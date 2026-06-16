#include "io/CellResampler.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vollenia {

namespace {

__device__ float clamp01(float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ int volumeIndex(int x, int y, int z, VolumeDesc desc)
{
    return (z * desc.ny + y) * desc.nx + x;
}

__device__ float sampleNearest(const float* source, VolumeDesc desc, float x, float y, float z)
{
    const int sx = min(max(static_cast<int>(floorf(x + 0.5f)), 0), desc.nx - 1);
    const int sy = min(max(static_cast<int>(floorf(y + 0.5f)), 0), desc.ny - 1);
    const int sz = min(max(static_cast<int>(floorf(z + 0.5f)), 0), desc.nz - 1);
    return source[volumeIndex(sx, sy, sz, desc)];
}

__device__ float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

__device__ float sampleTrilinear(const float* source, VolumeDesc desc, float x, float y, float z)
{
    const float cx = fminf(fmaxf(x, 0.0f), static_cast<float>(desc.nx - 1));
    const float cy = fminf(fmaxf(y, 0.0f), static_cast<float>(desc.ny - 1));
    const float cz = fminf(fmaxf(z, 0.0f), static_cast<float>(desc.nz - 1));

    const int x0 = static_cast<int>(floorf(cx));
    const int y0 = static_cast<int>(floorf(cy));
    const int z0 = static_cast<int>(floorf(cz));
    const int x1 = min(x0 + 1, desc.nx - 1);
    const int y1 = min(y0 + 1, desc.ny - 1);
    const int z1 = min(z0 + 1, desc.nz - 1);

    const float tx = cx - static_cast<float>(x0);
    const float ty = cy - static_cast<float>(y0);
    const float tz = cz - static_cast<float>(z0);

    const float c000 = source[volumeIndex(x0, y0, z0, desc)];
    const float c100 = source[volumeIndex(x1, y0, z0, desc)];
    const float c010 = source[volumeIndex(x0, y1, z0, desc)];
    const float c110 = source[volumeIndex(x1, y1, z0, desc)];
    const float c001 = source[volumeIndex(x0, y0, z1, desc)];
    const float c101 = source[volumeIndex(x1, y0, z1, desc)];
    const float c011 = source[volumeIndex(x0, y1, z1, desc)];
    const float c111 = source[volumeIndex(x1, y1, z1, desc)];

    const float c00 = lerp(c000, c100, tx);
    const float c10 = lerp(c010, c110, tx);
    const float c01 = lerp(c001, c101, tx);
    const float c11 = lerp(c011, c111, tx);
    const float c0 = lerp(c00, c10, ty);
    const float c1 = lerp(c01, c11, ty);
    return lerp(c0, c1, tz);
}

__global__ void resampleKernel(
    float* output,
    VolumeDesc output_desc,
    const float* source,
    VolumeDesc source_desc,
    float scale,
    CellResampleMode mode)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= output_desc.nx || y >= output_desc.ny || z >= output_desc.nz) {
        return;
    }

    const float inv_scale = 1.0f / fmaxf(scale, 1.0e-5f);
    const float sx = (static_cast<float>(x) + 0.5f) * inv_scale - 0.5f;
    const float sy = (static_cast<float>(y) + 0.5f) * inv_scale - 0.5f;
    const float sz = (static_cast<float>(z) + 0.5f) * inv_scale - 0.5f;

    const float value = mode == CellResampleMode::Nearest
        ? sampleNearest(source, source_desc, sx, sy, sz)
        : sampleTrilinear(source, source_desc, sx, sy, sz);
    output[volumeIndex(x, y, z, output_desc)] = clamp01(value);
}

int scaledDim(int dim, float scale)
{
    return std::max(1, static_cast<int>(std::lround(static_cast<float>(dim) * scale)));
}

} // namespace

const char* cellResampleModeName(CellResampleMode mode)
{
    switch (mode) {
    case CellResampleMode::Nearest:
        return "Nearest";
    case CellResampleMode::Trilinear:
        return "Trilinear";
    default:
        return "Unknown";
    }
}

void CellResampler::resampleToDevice(
    DeviceVolume& output,
    DeviceVolumeView source,
    float scale,
    CellResampleMode mode)
{
    if (source.data == nullptr || !isValidVolumeDesc(source.desc)) {
        throw std::runtime_error("Cannot resample invalid imported cells");
    }
    scale = std::max(scale, 1.0e-5f);
    const VolumeDesc output_desc {
        scaledDim(source.desc.nx, scale),
        scaledDim(source.desc.ny, scale),
        scaledDim(source.desc.nz, scale),
    };
    output.resize(output_desc);

    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(output_desc.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(output_desc.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(output_desc.nz) + block.z - 1U) / block.z);
    resampleKernel<<<grid, block>>>(
        output.data(),
        output_desc,
        source.data,
        source.desc,
        scale,
        mode);
    VOL_CUDA_CHECK(cudaGetLastError());
}

} // namespace vollenia
