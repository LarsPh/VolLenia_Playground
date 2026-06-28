#include "sim/ExpandedFlowSimulation.h"

#include "core/CudaCheck.h"
#include "sim/CufftCheck.h"
#include "sim/FlowTransport.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace vollenia {

namespace {

__device__ float clamp01(float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__global__ void seedModelSpecKernel(float* state, VolumeDesc desc, int channels, unsigned int seed)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= desc.nx || y >= desc.ny || z >= desc.nz) {
        return;
    }

    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(desc.nx);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(desc.ny);
    const float w = (static_cast<float>(z) + 0.5f) / static_cast<float>(desc.nz);
    const float px = 2.0f * u - 1.0f;
    const float py = 2.0f * v - 1.0f;
    const float pz = 2.0f * w - 1.0f;
    const std::size_t voxel = static_cast<std::size_t>((z * desc.ny + y) * desc.nx + x);
    const std::size_t count = static_cast<std::size_t>(desc.nx) * static_cast<std::size_t>(desc.ny) * static_cast<std::size_t>(desc.nz);

    for (int c = 0; c < channels; ++c) {
        const float offset = 0.12f * static_cast<float>(c);
        const float cx = -0.18f + offset;
        const float cy = 0.10f * sinf(static_cast<float>(seed % 17) + static_cast<float>(c));
        const float cz = 0.10f * cosf(static_cast<float>(seed % 11) + static_cast<float>(c));
        const float dx = px - cx;
        const float dy = py - cy;
        const float dz = pz - cz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float shell_r = sqrtf(px * px + py * py + pz * pz);
        float value = 0.85f * expf(-d2 / (2.0f * 0.18f * 0.18f));
        value += 0.18f * expf(-powf((shell_r - 0.42f) / 0.08f, 2.0f));
        state[static_cast<std::size_t>(c) * count + voxel] = clamp01(value);
    }
}

__global__ void multiplySpectrumKernel(
    cufftComplex* output,
    const cufftComplex* state,
    const cufftComplex* kernel,
    std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const cufftComplex a = state[index];
    const cufftComplex b = kernel[index];
    output[index] = cufftComplex {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ float growthValue(float u, GrowthSpec growth)
{
    const float sigma = fmaxf(growth.sigma, 1.0e-6f);
    const float diff = u - growth.mu;
    if (growth.family == GrowthFamily::Gaussian) {
        return 2.0f * expf(-0.5f * (diff * diff) / (sigma * sigma)) - 1.0f;
    }
    const float value = fmaxf(0.0f, 1.0f - (diff * diff) / (9.0f * sigma * sigma));
    const float value2 = value * value;
    return 2.0f * value2 * value2 - 1.0f;
}

__global__ void accumulateGrowthKernel(
    float* affinity,
    const float* potential,
    int* invalid_flag,
    std::size_t count,
    float inv_n,
    float weight,
    GrowthSpec growth)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float u = potential[index] * inv_n;
    float value = weight * growthValue(u, growth);
    if (!isfinite(value)) {
        atomicExch(invalid_flag, 1);
        value = 0.0f;
    }
    affinity[index] += value;
}

__global__ void applyExpandedAdditiveKernel(
    float* next_channel,
    const float* state_channel,
    const float* affinity_channel,
    int* invalid_flag,
    std::size_t count,
    float dt)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = state_channel[index] + dt * affinity_channel[index];
    if (!isfinite(value)) {
        atomicExch(invalid_flag, 1);
        value = 0.0f;
    }
    next_channel[index] = clamp01(value);
}

void swapVolumes(DeviceMultiVolume& a, DeviceMultiVolume& b)
{
    a.swap(b);
}

} // namespace

ExpandedFlowSimulation::~ExpandedFlowSimulation() noexcept
{
    destroyNoThrow();
}

void ExpandedFlowSimulation::initialize(VolumeDesc desc, const ModelSpec& spec)
{
    desc.nx = std::clamp(desc.nx, 16, 256);
    desc.ny = std::clamp(desc.ny, 16, 256);
    desc.nz = std::clamp(desc.nz, 16, 256);
    spec_ = spec;
    render_channel_ = std::clamp(spec_.render_channel, 0, spec_.channelCount() - 1);
    ensureBuffers(desc, spec_.channelCount());
    kernel_bank_.build(desc_, spec_, spec_.name);
    resetSeed(seed_);
    updateStatus("Lenia model initialized");
}

void ExpandedFlowSimulation::resetSeed(unsigned int seed)
{
    if (!state_.isValid()) {
        throw std::runtime_error("Cannot reset Lenia model seed before initialization");
    }
    seed_ = seed;
    state_.clear(0.0f);
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(desc_.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(desc_.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(desc_.nz) + block.z - 1U) / block.z);
    seedModelSpecKernel<<<grid, block>>>(state_.data(), desc_, channels_, seed_);
    VOL_CUDA_CHECK(cudaGetLastError());
    for (int c = 0; c < channels_; ++c) {
        if (!spec_.isMatterChannel(c)) {
            launchClearFloat(state_.channelData(c), voxel_count_, 0.0f);
        }
    }
    status_.generation = 0;
    status_.mass_after = matterMass();
    status_.mass_before = status_.mass_after;
    status_.mass_ratio = 1.0;
    updateStatus("Lenia model seed reset");
}

void ExpandedFlowSimulation::simulateSteps(int steps)
{
    if (steps <= 0) {
        return;
    }
    if (!isInitialized()) {
        throw std::runtime_error("Cannot step Lenia model simulation before initialization");
    }

    status_.mass_before = matterMass();
    VOL_CUDA_CHECK(cudaMemset(invalid_flag_, 0, sizeof(int)));
    for (int i = 0; i < steps; ++i) {
        computeAffinity();
        if (spec_.update_mode == ModelUpdateMode::Flow) {
            stepFlow();
        } else {
            stepExpandedAdditive();
        }
        swapVolumes(state_, next_);
        ++status_.generation;
    }

    int invalid = 0;
    VOL_CUDA_CHECK(cudaMemcpy(&invalid, invalid_flag_, sizeof(int), cudaMemcpyDeviceToHost));
    status_.last_step_had_invalid_values = invalid != 0;
    status_.mass_after = matterMass();
    status_.mass_ratio = status_.mass_before > 1.0e-9 ? status_.mass_after / status_.mass_before : 1.0;
    updateStatus(status_.last_step_had_invalid_values ? "Lenia model running; invalid values clamped" : "Lenia model running");
}

void ExpandedFlowSimulation::setRenderChannel(int channel)
{
    render_channel_ = std::clamp(channel, 0, std::max(channels_ - 1, 0));
    updateStatus("Lenia model render channel changed");
}

DeviceVolumeView ExpandedFlowSimulation::currentRenderView() const
{
    return state_.channelView(render_channel_);
}

DeviceVolumeView ExpandedFlowSimulation::channelView(int channel) const
{
    return state_.channelView(channel);
}

void ExpandedFlowSimulation::ensureBuffers(VolumeDesc desc, int channels)
{
    if (state_.isValid() && volumeDescEquals(desc_, desc) && channels_ == channels) {
        return;
    }

    destroy();
    desc_ = desc;
    channels_ = channels;
    voxel_count_ = volumeVoxelCount(desc_);
    spectrum_count_ = static_cast<std::size_t>(desc_.nz)
        * static_cast<std::size_t>(desc_.ny)
        * static_cast<std::size_t>(desc_.nx / 2 + 1);

    state_.resize(desc_, channels_);
    next_.resize(desc_, channels_);
    affinity_.resize(desc_, channels_);
    potential_.resize(desc_);
    matter_sum_.resize(desc_);

    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state_spectra_), spectrum_count_ * static_cast<std::size_t>(channels_) * sizeof(cufftComplex)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&work_spectrum_), spectrum_count_ * sizeof(cufftComplex)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_u_), voxel_count_ * sizeof(float3)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_a_sum_), voxel_count_ * sizeof(float3)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&flow_), voxel_count_ * static_cast<std::size_t>(channels_) * sizeof(float3)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&invalid_flag_), sizeof(int)));
    VOL_CUFFT_CHECK(cufftPlan3d(&r2c_plan_, desc_.nz, desc_.ny, desc_.nx, CUFFT_R2C));
    VOL_CUFFT_CHECK(cufftPlan3d(&c2r_plan_, desc_.nz, desc_.ny, desc_.nx, CUFFT_C2R));
}

void ExpandedFlowSimulation::computeAffinity()
{
    affinity_.clear(0.0f);
    const int block = 256;
    const int spectrum_grid = static_cast<int>((spectrum_count_ + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    const int voxel_grid = static_cast<int>((voxel_count_ + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    const float inv_n = 1.0f / static_cast<float>(voxel_count_);

    for (int c = 0; c < channels_; ++c) {
        VOL_CUFFT_CHECK(cufftExecR2C(r2c_plan_, state_.channelData(c), state_spectra_ + static_cast<std::size_t>(c) * spectrum_count_));
    }

    for (int k = 0; k < spec_.kernelCount(); ++k) {
        const KernelSpec& kernel = spec_.kernels[static_cast<std::size_t>(k)];
        if (!spec_.isMatterChannel(kernel.dst)) {
            continue;
        }
        multiplySpectrumKernel<<<spectrum_grid, block>>>(
            work_spectrum_,
            state_spectra_ + static_cast<std::size_t>(kernel.src) * spectrum_count_,
            kernel_bank_.spectrum(k),
            spectrum_count_);
        VOL_CUDA_CHECK(cudaGetLastError());
        VOL_CUFFT_CHECK(cufftExecC2R(c2r_plan_, work_spectrum_, potential_.data()));
        accumulateGrowthKernel<<<voxel_grid, block>>>(
            affinity_.channelData(kernel.dst),
            potential_.data(),
            invalid_flag_,
            voxel_count_,
            inv_n,
            kernel.weight,
            kernel.growth);
        VOL_CUDA_CHECK(cudaGetLastError());
    }
}

void ExpandedFlowSimulation::stepExpandedAdditive()
{
    const int block = 256;
    const int voxel_grid = static_cast<int>((voxel_count_ + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    for (int c = 0; c < channels_; ++c) {
        if (spec_.isMatterChannel(c)) {
            applyExpandedAdditiveKernel<<<voxel_grid, block>>>(
                next_.channelData(c),
                state_.channelData(c),
                affinity_.channelData(c),
                invalid_flag_,
                voxel_count_,
                spec_.dt);
            VOL_CUDA_CHECK(cudaGetLastError());
        } else {
            launchCopyFloat(next_.channelData(c), state_.channelData(c), voxel_count_);
        }
    }
}

void ExpandedFlowSimulation::stepFlow()
{
    matter_sum_.clear(0.0f);
    for (int c = 0; c < channels_; ++c) {
        if (spec_.isMatterChannel(c)) {
            launchAddScaled(matter_sum_.data(), state_.channelData(c), voxel_count_, 1.0f);
        }
    }
    launchSobelGradient3D(matter_sum_.data(), grad_a_sum_, desc_);

    for (int c = 0; c < channels_; ++c) {
        if (!spec_.isMatterChannel(c)) {
            launchCopyFloat(next_.channelData(c), state_.channelData(c), voxel_count_);
            continue;
        }
        launchSobelGradient3D(affinity_.channelData(c), grad_u_, desc_);
        float3* channel_flow = flow_ + static_cast<std::size_t>(c) * voxel_count_;
        launchComputeFlowField(
            channel_flow,
            grad_u_,
            grad_a_sum_,
            matter_sum_.data(),
            desc_,
            spec_.flow.theta_A,
            spec_.flow.alpha_power,
            spec_.flow.flow_max);
        launchFlowTransportSigmaHalf(
            next_.channelData(c),
            state_.channelData(c),
            channel_flow,
            desc_,
            spec_.dt,
            spec_.flow.flow_max,
            spec_.flow.border);
    }
}

double ExpandedFlowSimulation::matterMass() const
{
    if (!state_.isValid()) {
        return 0.0;
    }
    std::vector<float> host(voxel_count_);
    double total = 0.0;
    for (int c = 0; c < channels_; ++c) {
        if (!spec_.isMatterChannel(c)) {
            continue;
        }
        VOL_CUDA_CHECK(cudaMemcpy(host.data(), state_.channelData(c), voxel_count_ * sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : host) {
            if (std::isfinite(value)) {
                total += static_cast<double>(value);
            }
        }
    }
    return total;
}

void ExpandedFlowSimulation::updateStatus(const char* status_text)
{
    status_.initialized = state_.isValid();
    status_.desc = desc_;
    status_.byte_size = state_.byteSize();
    status_.channel_count = channels_;
    status_.kernel_count = spec_.kernelCount();
    status_.render_channel = render_channel_;
    status_.update_mode = modelUpdateModeName(spec_.update_mode);
    status_.model_name = spec_.name;
    status_.status = status_text;
    status_.last_error = status_.last_step_had_invalid_values ? "One or more NaN/Inf values were clamped during update." : "";
}

void ExpandedFlowSimulation::destroy()
{
    destroyPlans();
    kernel_bank_.destroy();
    if (state_spectra_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(state_spectra_));
        state_spectra_ = nullptr;
    }
    if (work_spectrum_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(work_spectrum_));
        work_spectrum_ = nullptr;
    }
    if (grad_u_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(grad_u_));
        grad_u_ = nullptr;
    }
    if (grad_a_sum_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(grad_a_sum_));
        grad_a_sum_ = nullptr;
    }
    if (flow_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(flow_));
        flow_ = nullptr;
    }
    if (invalid_flag_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(invalid_flag_));
        invalid_flag_ = nullptr;
    }
    state_.destroy();
    next_.destroy();
    affinity_.destroy();
    potential_.destroy();
    matter_sum_.destroy();
    desc_ = {};
    voxel_count_ = 0;
    spectrum_count_ = 0;
    channels_ = 0;
    render_channel_ = 0;
    status_ = {};
}

void ExpandedFlowSimulation::destroyPlans()
{
    if (r2c_plan_ != 0) {
        VOL_CUFFT_CHECK(cufftDestroy(r2c_plan_));
        r2c_plan_ = 0;
    }
    if (c2r_plan_ != 0) {
        VOL_CUFFT_CHECK(cufftDestroy(c2r_plan_));
        c2r_plan_ = 0;
    }
}

void ExpandedFlowSimulation::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy ExpandedFlow simulation cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
