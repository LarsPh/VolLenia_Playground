#include "sim/LeniaSeeds.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <stdexcept>

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

__device__ unsigned int hash32(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ float hashUnit(unsigned int x, unsigned int y, unsigned int z, unsigned int seed)
{
    const unsigned int mixed = hash32(x * 73856093U ^ y * 19349663U ^ z * 83492791U ^ seed * 2654435761U);
    return static_cast<float>(mixed & 0x00ffffffU) / static_cast<float>(0x01000000U);
}

__device__ float gaussian3(float3 p, float3 c, float radius)
{
    const float dx = p.x - c.x;
    const float dy = p.y - c.y;
    const float dz = p.z - c.z;
    const float d2 = dx * dx + dy * dy + dz * dz;
    return expf(-d2 / (2.0f * radius * radius));
}

__global__ void seedKernel(float* output, VolumeDesc desc, LeniaSeedPreset preset, unsigned int seed)
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
    const float random = hashUnit(static_cast<unsigned int>(x), static_cast<unsigned int>(y), static_cast<unsigned int>(z), seed);

    float value = 0.0f;
    if (preset == LeniaSeedPreset::ReferenceRandomBox) {
        const int bx = max(desc.nx / 3, 2);
        const int by = max(desc.ny / 3, 2);
        const int bz = max(desc.nz / 3, 2);
        const bool inside = abs(x - desc.nx / 2) <= bx / 2
            && abs(y - desc.ny / 2) <= by / 2
            && abs(z - desc.nz / 2) <= bz / 2;
        if (inside && random < 0.48f) {
            value = 0.18f + 0.82f * hashUnit(static_cast<unsigned int>(x + 17), static_cast<unsigned int>(y + 31), static_cast<unsigned int>(z + 47), seed);
        }
    } else if (preset == LeniaSeedPreset::CenteredRandomBall) {
        const float shell = 1.0f - smoothstep(0.36f, 0.48f, r);
        value = shell * (0.20f + 0.80f * random);
    } else if (preset == LeniaSeedPreset::AsymmetricGaussianCluster) {
        value += 0.95f * gaussian3(p, make_float3(-0.28f, -0.08f, 0.04f), 0.17f);
        value += 0.74f * gaussian3(p, make_float3(0.20f, 0.18f, -0.10f), 0.22f);
        value += 0.56f * gaussian3(p, make_float3(0.06f, -0.28f, 0.24f), 0.14f);
        value += 0.25f * random * (1.0f - smoothstep(0.44f, 0.70f, r));
    } else if (preset == LeniaSeedPreset::ShellInternalBlobs) {
        const float shell = expf(-powf((r - 0.48f) / 0.060f, 2.0f));
        const float blobs = 0.70f * gaussian3(p, make_float3(-0.16f, 0.08f, -0.10f), 0.16f)
            + 0.62f * gaussian3(p, make_float3(0.18f, -0.12f, 0.14f), 0.18f)
            + 0.44f * gaussian3(p, make_float3(0.02f, 0.24f, -0.22f), 0.13f);
        value = 0.45f * shell + blobs;
    } else {
        value = gaussian3(p, make_float3(0.0f, 0.0f, 0.0f), 0.080f)
            + 0.45f * gaussian3(p, make_float3(0.12f, -0.04f, 0.06f), 0.055f);
    }

    const int index = (z * desc.ny + y) * desc.nx + x;
    output[index] = clamp01(value);
}

} // namespace

void launchLeniaSeed(DeviceVolume& volume, LeniaSeedPreset preset, unsigned int seed)
{
    if (!volume.isValid()) {
        throw std::runtime_error("Cannot generate Lenia seed into invalid volume");
    }

    const VolumeDesc desc = volume.desc();
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(desc.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(desc.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(desc.nz) + block.z - 1U) / block.z);
    seedKernel<<<grid, block>>>(volume.data(), desc, preset, seed);
    VOL_CUDA_CHECK(cudaGetLastError());
}

} // namespace vollenia
