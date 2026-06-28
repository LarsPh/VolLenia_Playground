#include "sim/FlowTransport.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace vollenia {

namespace {

__device__ int wrapIndex(int value, int size)
{
    value %= size;
    return value < 0 ? value + size : value;
}

__device__ std::size_t voxelIndex(int x, int y, int z, VolumeDesc desc)
{
    return static_cast<std::size_t>((z * desc.ny + y) * desc.nx + x);
}

__device__ float sampleScalar(const float* input, int x, int y, int z, VolumeDesc desc)
{
    return input[voxelIndex(wrapIndex(x, desc.nx), wrapIndex(y, desc.ny), wrapIndex(z, desc.nz), desc)];
}

__device__ float clampFloat(float value, float lo, float hi)
{
    return fminf(fmaxf(value, lo), hi);
}

__global__ void clearFloatKernel(float* data, std::size_t count, float value)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    data[index] = value;
}

__global__ void copyFloatKernel(float* dst, const float* src, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    dst[index] = src[index];
}

__global__ void addScaledKernel(float* dst, const float* src, std::size_t count, float scale)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    dst[index] += scale * src[index];
}

__global__ void sobelGradientKernel(const float* input, float3* output, VolumeDesc desc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= desc.nx || y >= desc.ny || z >= desc.nz) {
        return;
    }

    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;
    for (int oz = -1; oz <= 1; ++oz) {
        const float wz = oz == 0 ? 2.0f : 1.0f;
        for (int oy = -1; oy <= 1; ++oy) {
            const float wy = oy == 0 ? 2.0f : 1.0f;
            for (int ox = -1; ox <= 1; ++ox) {
                const float wx = ox == 0 ? 2.0f : 1.0f;
                const float value = sampleScalar(input, x + ox, y + oy, z + oz, desc);
                gx += static_cast<float>(ox) * wy * wz * value;
                gy += static_cast<float>(oy) * wx * wz * value;
                gz += static_cast<float>(oz) * wx * wy * value;
            }
        }
    }
    output[voxelIndex(x, y, z, desc)] = make_float3(gx / 32.0f, gy / 32.0f, gz / 32.0f);
}

__global__ void computeFlowFieldKernel(
    float3* flow,
    const float3* grad_u,
    const float3* grad_a_sum,
    const float* a_sum,
    std::size_t count,
    float theta_a,
    float alpha_power,
    float flow_max)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float ratio = fmaxf(a_sum[index] / fmaxf(theta_a, 1.0e-6f), 0.0f);
    const float alpha = fminf(powf(ratio, alpha_power), 1.0f);
    float3 value = make_float3(
        (1.0f - alpha) * grad_u[index].x - alpha * grad_a_sum[index].x,
        (1.0f - alpha) * grad_u[index].y - alpha * grad_a_sum[index].y,
        (1.0f - alpha) * grad_u[index].z - alpha * grad_a_sum[index].z);
    value.x = clampFloat(value.x, -flow_max, flow_max);
    value.y = clampFloat(value.y, -flow_max, flow_max);
    value.z = clampFloat(value.z, -flow_max, flow_max);
    flow[index] = value;
}

__device__ float torusDistance(float a, float b, int size)
{
    const float direct = fabsf(a - b);
    return fminf(direct, static_cast<float>(size) - direct);
}

__global__ void flowTransportSigmaHalfKernel(
    float* next_channel,
    const float* state_channel,
    const float3* flow_channel,
    VolumeDesc desc,
    float dt,
    float flow_max,
    int dd,
    FlowBorder border)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= desc.nx || y >= desc.ny || z >= desc.nz) {
        return;
    }

    const float target_x = static_cast<float>(x) + 0.5f;
    const float target_y = static_cast<float>(y) + 0.5f;
    const float target_z = static_cast<float>(z) + 0.5f;
    const float ma = fmaxf(static_cast<float>(dd) - 0.5f, 0.0f);
    float total = 0.0f;

    for (int oz = -dd; oz <= dd; ++oz) {
        for (int oy = -dd; oy <= dd; ++oy) {
            for (int ox = -dd; ox <= dd; ++ox) {
                int sx = x - ox;
                int sy = y - oy;
                int sz = z - oz;
                if (border == FlowBorder::Torus) {
                    sx = wrapIndex(sx, desc.nx);
                    sy = wrapIndex(sy, desc.ny);
                    sz = wrapIndex(sz, desc.nz);
                } else if (sx < 0 || sx >= desc.nx || sy < 0 || sy >= desc.ny || sz < 0 || sz >= desc.nz) {
                    continue;
                }

                const std::size_t source_index = voxelIndex(sx, sy, sz, desc);
                const float3 f = flow_channel[source_index];
                float mu_x = static_cast<float>(sx) + 0.5f + clampFloat(dt * f.x, -ma, ma);
                float mu_y = static_cast<float>(sy) + 0.5f + clampFloat(dt * f.y, -ma, ma);
                float mu_z = static_cast<float>(sz) + 0.5f + clampFloat(dt * f.z, -ma, ma);
                if (border == FlowBorder::Wall) {
                    mu_x = clampFloat(mu_x, 0.5f, static_cast<float>(desc.nx) - 0.5f);
                    mu_y = clampFloat(mu_y, 0.5f, static_cast<float>(desc.ny) - 0.5f);
                    mu_z = clampFloat(mu_z, 0.5f, static_cast<float>(desc.nz) - 0.5f);
                }

                const float dx = border == FlowBorder::Torus ? torusDistance(target_x, mu_x, desc.nx) : fabsf(target_x - mu_x);
                const float dy = border == FlowBorder::Torus ? torusDistance(target_y, mu_y, desc.ny) : fabsf(target_y - mu_y);
                const float dz = border == FlowBorder::Torus ? torusDistance(target_z, mu_z, desc.nz) : fabsf(target_z - mu_z);
                const float weight = fmaxf(1.0f - dx, 0.0f) * fmaxf(1.0f - dy, 0.0f) * fmaxf(1.0f - dz, 0.0f);
                total += state_channel[source_index] * weight;
            }
        }
    }
    next_channel[voxelIndex(x, y, z, desc)] = fmaxf(total, 0.0f);
}

} // namespace

void launchClearFloat(float* data, std::size_t count, float value)
{
    const int block = 256;
    const int grid = static_cast<int>((count + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    clearFloatKernel<<<grid, block>>>(data, count, value);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void launchCopyFloat(float* dst, const float* src, std::size_t count)
{
    const int block = 256;
    const int grid = static_cast<int>((count + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    copyFloatKernel<<<grid, block>>>(dst, src, count);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void launchAddScaled(float* dst, const float* src, std::size_t count, float scale)
{
    const int block = 256;
    const int grid = static_cast<int>((count + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    addScaledKernel<<<grid, block>>>(dst, src, count, scale);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void launchSobelGradient3D(const float* input, float3* output, VolumeDesc desc)
{
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(desc.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(desc.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(desc.nz) + block.z - 1U) / block.z);
    sobelGradientKernel<<<grid, block>>>(input, output, desc);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void launchComputeFlowField(
    float3* flow,
    const float3* grad_u,
    const float3* grad_a_sum,
    const float* a_sum,
    VolumeDesc desc,
    float theta_a,
    float alpha_power,
    float flow_max)
{
    const std::size_t count = volumeVoxelCount(desc);
    const int block = 256;
    const int grid = static_cast<int>((count + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    computeFlowFieldKernel<<<grid, block>>>(flow, grad_u, grad_a_sum, a_sum, count, theta_a, alpha_power, flow_max);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void launchFlowTransportSigmaHalf(
    float* next_channel,
    const float* state_channel,
    const float3* flow_channel,
    VolumeDesc desc,
    float dt,
    float flow_max,
    FlowBorder border)
{
    const int dd = std::max(1, static_cast<int>(std::ceil(flow_max + 1.0e-6f)));
    const dim3 block(8, 8, 8);
    const dim3 grid(
        (static_cast<unsigned int>(desc.nx) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(desc.ny) + block.y - 1U) / block.y,
        (static_cast<unsigned int>(desc.nz) + block.z - 1U) / block.z);
    flowTransportSigmaHalfKernel<<<grid, block>>>(next_channel, state_channel, flow_channel, desc, dt, flow_max, dd, border);
    VOL_CUDA_CHECK(cudaGetLastError());
}

} // namespace vollenia
