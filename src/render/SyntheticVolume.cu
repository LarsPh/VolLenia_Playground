#include "render/SyntheticVolume.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace vollenia {

namespace {

__device__ float clamp01(float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ float smoothstep(float edge0, float edge1, float x)
{
    const float t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

__device__ float gaussian3(float3 p, float3 c, float radius)
{
    const float dx = p.x - c.x;
    const float dy = p.y - c.y;
    const float dz = p.z - c.z;
    const float d2 = dx * dx + dy * dy + dz * dz;
    return expf(-d2 / (2.0f * radius * radius));
}

__global__ void generateVolumeKernel(float* output, VolumeDesc desc, VolumePreset preset, unsigned int seed)
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
    const float3 p = make_float3(2.0f * u - 1.0f, 2.0f * v - 1.0f, 2.0f * w - 1.0f);
    const float r = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);

    float density = 0.0f;
    if (preset == VolumePreset::Sphere) {
        density = 1.0f - smoothstep(0.48f, 0.62f, r);
    } else if (preset == VolumePreset::Shell) {
        density = expf(-powf((r - 0.56f) / 0.055f, 2.0f));
        density += 0.25f * expf(-powf((r - 0.28f) / 0.04f, 2.0f));
    } else if (preset == VolumePreset::GaussianBlobs) {
        density += 0.95f * gaussian3(p, make_float3(-0.32f, -0.10f, -0.08f), 0.20f);
        density += 0.80f * gaussian3(p, make_float3(0.26f, 0.22f, 0.14f), 0.24f);
        density += 0.60f * gaussian3(p, make_float3(0.05f, -0.34f, 0.30f), 0.16f);
        density += 0.50f * gaussian3(p, make_float3(-0.10f, 0.32f, -0.28f), 0.18f);
    } else if (preset == VolumePreset::LeniaPhantom) {
        const float shell = expf(-powf((r - 0.58f) / 0.06f, 2.0f));
        const float inner = 0.75f * gaussian3(p, make_float3(-0.18f, 0.04f, -0.08f), 0.18f)
            + 0.65f * gaussian3(p, make_float3(0.22f, -0.10f, 0.16f), 0.20f)
            + 0.42f * gaussian3(p, make_float3(0.02f, 0.24f, -0.24f), 0.15f);
        const float bands = 0.5f + 0.5f * sinf(18.0f * r + 6.0f * p.x - 4.0f * p.z + static_cast<float>(seed % 17));
        const float wisps = smoothstep(0.16f, 0.72f, r) * (1.0f - smoothstep(0.74f, 0.95f, r)) * bands;
        density = 0.58f * shell + inner + 0.28f * wisps;
    } else {
        density = 0.15f + 0.85f * (0.55f * u + 0.30f * v + 0.15f * w);
    }

    const int index = (z * desc.ny + y) * desc.nx + x;
    output[index] = clamp01(density);
}

} // namespace

SyntheticVolume::~SyntheticVolume() noexcept
{
    destroyNoThrow();
}

void SyntheticVolume::resize(VolumeDesc desc)
{
    desc.nx = std::max(desc.nx, 1);
    desc.ny = std::max(desc.ny, 1);
    desc.nz = std::max(desc.nz, 1);

    if (data_ != nullptr && desc.nx == desc_.nx && desc.ny == desc_.ny && desc.nz == desc_.nz) {
        return;
    }

    destroy();
    desc_ = desc;
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), byteSize()));
}

void SyntheticVolume::generate(VolumePreset preset, unsigned int seed)
{
    if (!isValid()) {
        throw std::runtime_error("Cannot generate invalid synthetic volume");
    }

    preset_ = preset;
    seed_ = seed;

    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(desc_.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(desc_.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(desc_.nz) + block.z - 1U) / block.z);
    generateVolumeKernel<<<grid, block>>>(data_, desc_, preset_, seed_);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void SyntheticVolume::destroy()
{
    if (data_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
    }
    desc_ = {};
}

std::size_t SyntheticVolume::voxelCount() const
{
    return static_cast<std::size_t>(desc_.nx) * static_cast<std::size_t>(desc_.ny) * static_cast<std::size_t>(desc_.nz);
}

void SyntheticVolume::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy synthetic volume cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
