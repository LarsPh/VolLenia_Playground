#include "sim/LeniaSimulation.h"

#include "core/CudaCheck.h"
#include "sim/CufftCheck.h"
#include "sim/LeniaSeeds.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace vollenia {

namespace {

constexpr float kMinSigma = 1.0e-5f;
constexpr float kMinT = 1.0f;

LeniaParams sanitizeParams(LeniaParams params)
{
    params.radius = std::max(params.radius, 1.0f);
    params.T = std::max(params.T, kMinT);
    params.sigma = std::max(params.sigma, kMinSigma);
    params.shell_count = std::clamp(params.shell_count, 1, static_cast<int>(params.shell_weights.size()));
    bool has_non_zero_weight = false;
    for (int i = 0; i < params.shell_count; ++i) {
        has_non_zero_weight = has_non_zero_weight || params.shell_weights[static_cast<std::size_t>(i)] != 0.0f;
    }
    if (!has_non_zero_weight) {
        params.shell_weights[0] = 1.0f;
        params.shell_count = 1;
    }
    return params;
}

__device__ float clamp01(float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__global__ void multiplySpectrumKernel(
    cufftComplex* output,
    const cufftComplex* input,
    const cufftComplex* kernel,
    std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const cufftComplex a = input[index];
    const cufftComplex b = kernel[index];
    cufftComplex value;
    value.x = a.x * b.x - a.y * b.y;
    value.y = a.x * b.y + a.y * b.x;
    output[index] = value;
}

__global__ void updateStateKernel(
    float* state,
    const float* potential,
    int* invalid_flag,
    std::size_t count,
    float inv_n,
    float mu,
    float sigma,
    float T,
    GrowthFunctionType growth_function,
    bool validate_nan_inf)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const float u = potential[index] * inv_n;
    const float sigma_safe = fmaxf(sigma, 1.0e-5f);
    const float diff = u - mu;
    float growth = 0.0f;
    if (growth_function == GrowthFunctionType::Gaussian) {
        growth = 2.0f * expf(-(diff * diff) / (2.0f * sigma_safe * sigma_safe)) - 1.0f;
    } else if (growth_function == GrowthFunctionType::Step) {
        growth = fabsf(diff) <= sigma_safe ? 1.0f : -1.0f;
    } else {
        const float value = fmaxf(0.0f, 1.0f - (diff * diff) / (9.0f * sigma_safe * sigma_safe));
        const float value2 = value * value;
        growth = 2.0f * value2 * value2 - 1.0f;
    }
    float next = state[index] + growth / fmaxf(T, 1.0f);
    if (validate_nan_inf && !isfinite(next)) {
        atomicExch(invalid_flag, 1);
        next = 0.0f;
    }
    state[index] = clamp01(next);
}

__global__ void copyCenteredKernel(float* target, VolumeDesc target_desc, const float* source, VolumeDesc source_desc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= source_desc.nx || y >= source_desc.ny || z >= source_desc.nz) {
        return;
    }

    const int target_x = x + (target_desc.nx - source_desc.nx) / 2;
    const int target_y = y + (target_desc.ny - source_desc.ny) / 2;
    const int target_z = z + (target_desc.nz - source_desc.nz) / 2;
    if (target_x < 0 || target_x >= target_desc.nx || target_y < 0 || target_y >= target_desc.ny || target_z < 0 || target_z >= target_desc.nz) {
        return;
    }

    const int source_index = (z * source_desc.ny + y) * source_desc.nx + x;
    const int target_index = (target_z * target_desc.ny + target_y) * target_desc.nx + target_x;
    target[target_index] = clamp01(source[source_index]);
}

float kernelCoreValue(float r, KernelCoreType type)
{
    if (r < 0.0f || r > 1.0f) {
        return 0.0f;
    }
    if (type == KernelCoreType::ExponentialBump) {
        if (r <= 0.0f || r >= 1.0f) {
            return 0.0f;
        }
        return std::exp(4.0f - 1.0f / (r * (1.0f - r)));
    }
    if (type == KernelCoreType::Step) {
        return (r >= 0.25f && r <= 0.75f) ? 1.0f : 0.0f;
    }
    if (type == KernelCoreType::Staircase) {
        return (r >= 0.25f && r <= 0.75f) ? 1.0f : (r < 0.25f ? 0.5f : 0.0f);
    }
    const float value = 4.0f * r * (1.0f - r);
    return value * value * value * value;
}

float* allocateHostFloatBuffer(std::size_t count)
{
    float* pointer = static_cast<float*>(std::malloc(count * sizeof(float)));
    if (pointer == nullptr) {
        throw std::runtime_error("Failed to allocate host buffer for Lenia kernel");
    }
    return pointer;
}

void buildSpatialKernel(VolumeDesc desc, const LeniaParams& params, float* kernel)
{
    const std::size_t voxel_count = volumeVoxelCount(desc);
    std::fill(kernel, kernel + voxel_count, 0.0f);
    const int shell_count = params.shell_count;
    double sum = 0.0;

    for (int z = 0; z < desc.nz; ++z) {
        const int dz = std::min(z, desc.nz - z);
        for (int y = 0; y < desc.ny; ++y) {
            const int dy = std::min(y, desc.ny - y);
            for (int x = 0; x < desc.nx; ++x) {
                const int dx = std::min(x, desc.nx - x);
                const float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz));
                const float q = distance / params.radius;
                float value = 0.0f;
                if (q < 1.0f) {
                    const float shell_position = q * static_cast<float>(shell_count);
                    const int shell_index = std::clamp(static_cast<int>(std::floor(shell_position)), 0, shell_count - 1);
                    const float local_r = shell_position - std::floor(shell_position);
                    value = params.shell_weights[static_cast<std::size_t>(shell_index)] * kernelCoreValue(local_r, params.kernel_core);
                }
                const std::size_t index = static_cast<std::size_t>((z * desc.ny + y) * desc.nx + x);
                kernel[index] = value;
                sum += static_cast<double>(value);
            }
        }
    }

    if (sum <= 0.0) {
        throw std::runtime_error("Lenia kernel sum is zero; cannot normalize");
    }

    const double inv_sum = 1.0 / sum;
    for (std::size_t i = 0; i < voxel_count; ++i) {
        kernel[i] = static_cast<float>(static_cast<double>(kernel[i]) * inv_sum);
    }
}

} // namespace

LeniaSimulation::~LeniaSimulation() noexcept
{
    destroyNoThrow();
}

void LeniaSimulation::initialize(VolumeDesc desc, const LeniaParams& params, LeniaSeedPreset seed_preset, unsigned int seed)
{
    ensureBuffers(desc);
    params_ = sanitizeParams(params);
    seed_preset_ = seed_preset;
    seed_ = seed;
    kernel_dirty_ = true;
    rebuildKernel();
    resetSeed(seed_preset_, seed_);
    status_.simulation_status = "Initialized";
    status_.last_error = "";
    updateStatus();
}

void LeniaSimulation::resetSeed(LeniaSeedPreset seed_preset, unsigned int seed)
{
    if (!buffers_ready_) {
        throw std::runtime_error("Cannot reset Lenia seed before buffers are initialized");
    }

    seed_preset_ = seed_preset;
    seed_ = seed;
    launchLeniaSeed(state_, seed_preset_, seed_);
    status_.generation = 0;
    status_.simulation_status = "Seed reset";
    status_.last_step_had_invalid_values = false;
    status_.last_error = "";
    updateStatus();
}

void LeniaSimulation::resetImportedCells(DeviceVolumeView source_cells)
{
    if (!buffers_ready_) {
        throw std::runtime_error("Cannot reset imported Lenia cells before buffers are initialized");
    }
    if (source_cells.data == nullptr || !isValidVolumeDesc(source_cells.desc)) {
        throw std::runtime_error("Cannot reset imported Lenia cells from invalid source");
    }

    state_.clear(0.0f);
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(source_cells.desc.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(source_cells.desc.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(source_cells.desc.nz) + block.z - 1U) / block.z);
    copyCenteredKernel<<<grid, block>>>(state_.data(), desc_, source_cells.data, source_cells.desc);
    VOL_CUDA_CHECK(cudaGetLastError());

    status_.generation = 0;
    status_.simulation_status = "Imported cells reset";
    status_.last_step_had_invalid_values = false;
    status_.last_error = "";
    updateStatus();
}

void LeniaSimulation::setParams(const LeniaParams& params)
{
    const LeniaParams sanitized = sanitizeParams(params);
    if (!leniaKernelParamsEqual(params_, sanitized)) {
        kernel_dirty_ = true;
        status_.kernel_status = "Dirty";
    }
    params_ = sanitized;
}

void LeniaSimulation::rebuildKernel()
{
    if (!buffers_ready_) {
        throw std::runtime_error("Cannot rebuild Lenia kernel before buffers are initialized");
    }

    float* spatial = allocateHostFloatBuffer(spatial_kernel_.voxelCount());
    try {
        buildSpatialKernel(desc_, params_, spatial);
        VOL_CUDA_CHECK(cudaMemcpy(spatial_kernel_.data(), spatial, spatial_kernel_.byteSize(), cudaMemcpyHostToDevice));
    } catch (...) {
        std::free(spatial);
        throw;
    }
    std::free(spatial);
    VOL_CUFFT_CHECK(cufftExecR2C(r2c_plan_, spatial_kernel_.data(), kernel_spectrum_));
    kernel_dirty_ = false;
    status_.kernel_ready = true;
    status_.kernel_status = "Built";
    status_.last_error = "";
    VOL_CUDA_CHECK(cudaGetLastError());
}

void LeniaSimulation::simulateSteps(int steps, bool validate_nan_inf)
{
    if (steps <= 0) {
        return;
    }
    if (!buffers_ready_) {
        throw std::runtime_error("Cannot step Lenia simulation before initialization");
    }
    if (kernel_dirty_) {
        rebuildKernel();
    }

    const std::size_t voxel_count = state_.voxelCount();
    const int block = 256;
    const int spectrum_grid = static_cast<int>((spectrum_count_ + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    const int voxel_grid = static_cast<int>((voxel_count + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    const float inv_n = 1.0f / static_cast<float>(voxel_count);

    bool invalid_seen = false;
    for (int step = 0; step < steps; ++step) {
        VOL_CUDA_CHECK(cudaMemset(invalid_flag_, 0, sizeof(int)));
        VOL_CUFFT_CHECK(cufftExecR2C(r2c_plan_, state_.data(), state_spectrum_));
        multiplySpectrumKernel<<<spectrum_grid, block>>>(potential_spectrum_, state_spectrum_, kernel_spectrum_, spectrum_count_);
        VOL_CUDA_CHECK(cudaGetLastError());
        VOL_CUFFT_CHECK(cufftExecC2R(c2r_plan_, potential_spectrum_, potential_.data()));
        updateStateKernel<<<voxel_grid, block>>>(
            state_.data(),
            potential_.data(),
            invalid_flag_,
            voxel_count,
            inv_n,
            params_.mu,
            params_.sigma,
            params_.T,
            params_.growth_function,
            validate_nan_inf);
        VOL_CUDA_CHECK(cudaGetLastError());

        if (validate_nan_inf) {
            int invalid = 0;
            VOL_CUDA_CHECK(cudaMemcpy(&invalid, invalid_flag_, sizeof(int), cudaMemcpyDefault));
            invalid_seen = invalid_seen || invalid != 0;
        }
        ++status_.generation;
    }

    status_.last_step_had_invalid_values = invalid_seen;
    status_.simulation_status = invalid_seen ? "Running; invalid values clamped" : "Running";
    status_.last_error = invalid_seen ? "One or more NaN/Inf values were clamped during update." : "";
    updateStatus();
}

void LeniaSimulation::ensureBuffers(VolumeDesc desc)
{
    desc.nx = std::clamp(desc.nx, 1, 256);
    desc.ny = std::clamp(desc.ny, 1, 256);
    desc.nz = std::clamp(desc.nz, 1, 256);

    if (buffers_ready_ && volumeDescEquals(desc_, desc)) {
        return;
    }

    destroy();
    desc_ = desc;
    state_.resize(desc_);
    potential_.resize(desc_);
    spatial_kernel_.resize(desc_);

    spectrum_count_ = static_cast<std::size_t>(desc_.nz)
        * static_cast<std::size_t>(desc_.ny)
        * static_cast<std::size_t>(desc_.nx / 2 + 1);

    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state_spectrum_), spectrum_count_ * sizeof(cufftComplex)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&kernel_spectrum_), spectrum_count_ * sizeof(cufftComplex)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&potential_spectrum_), spectrum_count_ * sizeof(cufftComplex)));
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&invalid_flag_), sizeof(int)));
    VOL_CUFFT_CHECK(cufftPlan3d(&r2c_plan_, desc_.nz, desc_.ny, desc_.nx, CUFFT_R2C));
    VOL_CUFFT_CHECK(cufftPlan3d(&c2r_plan_, desc_.nz, desc_.ny, desc_.nx, CUFFT_C2R));

    buffers_ready_ = true;
    kernel_dirty_ = true;
    status_.initialized = true;
    status_.kernel_ready = false;
    status_.kernel_status = "Dirty";
    updateStatus();
}

void LeniaSimulation::destroy()
{
    destroyPlans();
    if (state_spectrum_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(state_spectrum_));
        state_spectrum_ = nullptr;
    }
    if (kernel_spectrum_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(kernel_spectrum_));
        kernel_spectrum_ = nullptr;
    }
    if (potential_spectrum_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(potential_spectrum_));
        potential_spectrum_ = nullptr;
    }
    if (invalid_flag_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(invalid_flag_));
        invalid_flag_ = nullptr;
    }
    state_.destroy();
    potential_.destroy();
    spatial_kernel_.destroy();
    desc_ = {};
    spectrum_count_ = 0;
    buffers_ready_ = false;
    kernel_dirty_ = true;
    status_ = {};
}

void LeniaSimulation::destroyPlans()
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

void LeniaSimulation::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy Lenia simulation cleanly: " << exception.what() << '\n';
    }
}

void LeniaSimulation::updateStatus()
{
    status_.desc = desc_;
    status_.byte_size = state_.byteSize();
    status_.initialized = buffers_ready_ && state_.isValid();
    status_.kernel_ready = status_.kernel_ready && !kernel_dirty_;
}

} // namespace vollenia
