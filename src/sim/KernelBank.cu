#include "sim/KernelBank.h"

#include "core/CudaCheck.h"
#include "sim/CufftCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace vollenia {

namespace {

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

float legacyCoreValue(float q, const KernelSpec& kernel)
{
    if (q < 0.0f || q >= 1.0f) {
        return 0.0f;
    }
    const std::vector<float>& weights = kernel.shell_weights;
    if (weights.empty()) {
        const float value = 4.0f * q * (1.0f - q);
        return value * value * value * value;
    }
    const float shell_position = q * static_cast<float>(weights.size());
    const int shell_index = std::clamp(static_cast<int>(std::floor(shell_position)), 0, static_cast<int>(weights.size()) - 1);
    const float local = shell_position - std::floor(shell_position);
    const float value = 4.0f * local * (1.0f - local);
    return weights[static_cast<std::size_t>(shell_index)] * value * value * value * value;
}

float smoothKernelValue(float q, const KernelSpec& kernel)
{
    const float envelope = sigmoid(kernel.envelope_sharpness * (1.0f - q));
    double raw = 0.0;
    for (const SmoothKernelBasis& basis : kernel.basis) {
        const float width = std::max(basis.width, 1.0e-6f);
        const float diff = (q - basis.center) / width;
        raw += static_cast<double>(basis.amplitude) * std::exp(-0.5 * static_cast<double>(diff * diff));
    }
    return envelope * static_cast<float>(raw);
}

float kernelValueForDistance(float distance, const KernelSpec& kernel)
{
    const float radius = std::max(kernel.radius, 1.0e-6f);
    const float q = distance / radius;
    if (kernel.family == KernelFamily::LegacyShell) {
        return legacyCoreValue(q, kernel);
    }
    return smoothKernelValue(q, kernel);
}

std::vector<float> buildKernelHost(VolumeDesc desc, const KernelSpec& kernel, bool centered)
{
    std::vector<float> values(volumeVoxelCount(desc), 0.0f);
    double sum = 0.0;
    for (int z = 0; z < desc.nz; ++z) {
        const int dz = centered ? z - desc.nz / 2 : std::min(z, desc.nz - z);
        for (int y = 0; y < desc.ny; ++y) {
            const int dy = centered ? y - desc.ny / 2 : std::min(y, desc.ny - y);
            for (int x = 0; x < desc.nx; ++x) {
                const int dx = centered ? x - desc.nx / 2 : std::min(x, desc.nx - x);
                const float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz));
                const float value = kernelValueForDistance(distance, kernel);
                const std::size_t index = static_cast<std::size_t>((z * desc.ny + y) * desc.nx + x);
                values[index] = value;
                sum += static_cast<double>(value);
            }
        }
    }
    if (sum <= 0.0) {
        throw std::runtime_error("ModelSpec kernel sum is zero and cannot be normalized: " + kernel.name);
    }
    const double inv_sum = 1.0 / sum;
    for (float& value : values) {
        value = static_cast<float>(static_cast<double>(value) * inv_sum);
    }
    return values;
}

std::string sanitizeFilePart(std::string value)
{
    for (char& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return value.empty() ? "model" : value;
}

} // namespace

KernelBank::~KernelBank() noexcept
{
    destroyNoThrow();
}

void KernelBank::build(VolumeDesc desc, const ModelSpec& spec, const std::string& debug_name)
{
    destroy();
    desc_ = desc;
    kernel_count_ = spec.kernelCount();
    spectrum_count_ = static_cast<std::size_t>(desc_.nz)
        * static_cast<std::size_t>(desc_.ny)
        * static_cast<std::size_t>(desc_.nx / 2 + 1);

    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&spectra_), spectrum_count_ * static_cast<std::size_t>(kernel_count_) * sizeof(cufftComplex)));
    VOL_CUFFT_CHECK(cufftPlan3d(&r2c_plan_, desc_.nz, desc_.ny, desc_.nx, CUFFT_R2C));

    float* device_kernel = nullptr;
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_kernel), volumeByteSize(desc_)));
    try {
        const std::filesystem::path output_dir("outputs/kernel_debug");
        for (int k = 0; k < kernel_count_; ++k) {
            const KernelSpec& kernel = spec.kernels[static_cast<std::size_t>(k)];
            const std::vector<float> spatial = buildKernelHost(desc_, kernel, false);
            dumpKernelDebug(output_dir, debug_name, kernel, desc_, spatial);
            VOL_CUDA_CHECK(cudaMemcpy(device_kernel, spatial.data(), volumeByteSize(desc_), cudaMemcpyHostToDevice));
            VOL_CUFFT_CHECK(cufftExecR2C(r2c_plan_, device_kernel, spectra_ + static_cast<std::size_t>(k) * spectrum_count_));
        }
    } catch (...) {
        cudaFree(device_kernel);
        throw;
    }
    VOL_CUDA_CHECK(cudaFree(device_kernel));
}

const cufftComplex* KernelBank::spectrum(int kernel_index) const
{
    if (!isValid() || kernel_index < 0 || kernel_index >= kernel_count_) {
        throw std::out_of_range("KernelBank kernel index out of range");
    }
    return spectra_ + static_cast<std::size_t>(kernel_index) * spectrum_count_;
}

void KernelBank::dumpKernelDebug(
    const std::filesystem::path& output_dir,
    const std::string& debug_name,
    const KernelSpec& kernel,
    VolumeDesc desc,
    const std::vector<float>& fft_origin_kernel) const
{
    std::filesystem::create_directories(output_dir);
    const std::string stem = sanitizeFilePart(debug_name) + "_" + sanitizeFilePart(kernel.name);
    const std::filesystem::path profile_path = output_dir / (stem + "_profile.csv");
    const std::filesystem::path slice_path = output_dir / (stem + "_slice_zmid.f32");

    const int max_radius = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(desc.nx * desc.nx + desc.ny * desc.ny + desc.nz * desc.nz))));
    std::vector<double> sums(static_cast<std::size_t>(max_radius + 1), 0.0);
    std::vector<float> max_values(static_cast<std::size_t>(max_radius + 1), 0.0f);
    std::vector<int> counts(static_cast<std::size_t>(max_radius + 1), 0);
    for (int z = 0; z < desc.nz; ++z) {
        const int dz = std::min(z, desc.nz - z);
        for (int y = 0; y < desc.ny; ++y) {
            const int dy = std::min(y, desc.ny - y);
            for (int x = 0; x < desc.nx; ++x) {
                const int dx = std::min(x, desc.nx - x);
                const int bin = static_cast<int>(std::floor(std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz))));
                const std::size_t index = static_cast<std::size_t>((z * desc.ny + y) * desc.nx + x);
                sums[static_cast<std::size_t>(bin)] += fft_origin_kernel[index];
                max_values[static_cast<std::size_t>(bin)] = std::max(max_values[static_cast<std::size_t>(bin)], fft_origin_kernel[index]);
                ++counts[static_cast<std::size_t>(bin)];
            }
        }
    }

    std::ofstream profile(profile_path);
    profile << "radius_bin,mean_value,max_value,sample_count\n";
    for (int bin = 0; bin <= max_radius; ++bin) {
        if (counts[static_cast<std::size_t>(bin)] == 0) {
            continue;
        }
        profile << bin << ','
                << (sums[static_cast<std::size_t>(bin)] / static_cast<double>(counts[static_cast<std::size_t>(bin)])) << ','
                << max_values[static_cast<std::size_t>(bin)] << ','
                << counts[static_cast<std::size_t>(bin)] << '\n';
    }

    const std::vector<float> centered = buildKernelHost(desc, kernel, true);
    std::ofstream slice(slice_path, std::ios::binary);
    const std::size_t z_offset = static_cast<std::size_t>(desc.nz / 2) * static_cast<std::size_t>(desc.ny) * static_cast<std::size_t>(desc.nx);
    slice.write(
        reinterpret_cast<const char*>(centered.data() + z_offset),
        static_cast<std::streamsize>(static_cast<std::size_t>(desc.nx) * static_cast<std::size_t>(desc.ny) * sizeof(float)));
}

void KernelBank::destroy()
{
    if (r2c_plan_ != 0) {
        VOL_CUFFT_CHECK(cufftDestroy(r2c_plan_));
        r2c_plan_ = 0;
    }
    if (spectra_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(spectra_));
        spectra_ = nullptr;
    }
    desc_ = {};
    kernel_count_ = 0;
    spectrum_count_ = 0;
}

void KernelBank::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy kernel bank cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
