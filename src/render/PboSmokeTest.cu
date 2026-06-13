#include "render/PboSmokeTest.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

namespace vollenia {

namespace {

__device__ unsigned char floatToByte(float value)
{
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<unsigned char>(255.0f * value + 0.5f);
}

__global__ void pboSmokeKernel(uchar4* output, int width, int height, float time_seconds)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    const float dx = u - 0.5f;
    const float dy = v - 0.5f;
    const float radius = sqrtf(dx * dx + dy * dy);
    const float wave = 0.5f + 0.5f * sinf(42.0f * radius - 3.5f * time_seconds);
    const float sweep = 0.5f + 0.5f * sinf(6.2831853f * (u + 0.15f * time_seconds));

    uchar4 color;
    color.x = floatToByte(0.15f + 0.85f * u);
    color.y = floatToByte(0.10f + 0.65f * v + 0.25f * wave);
    color.z = floatToByte(0.20f + 0.55f * wave + 0.25f * sweep);
    color.w = 255;

    output[y * width + x] = color;
}

} // namespace

void launchPboSmokeTest(uchar4* output, int width, int height, float time_seconds)
{
    if (output == nullptr || width <= 0 || height <= 0) {
        return;
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1U) / block.y);

    pboSmokeKernel<<<grid, block>>>(output, width, height, time_seconds);
    VOL_CUDA_CHECK(cudaGetLastError());

#ifndef NDEBUG
    VOL_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

} // namespace vollenia
