#include "sim/DeviceVolume.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

namespace vollenia {

namespace {

__global__ void fillKernel(float* data, std::size_t count, float value)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    data[index] = value;
}

} // namespace

DeviceVolume::~DeviceVolume() noexcept
{
    destroyNoThrow();
}

void DeviceVolume::resize(VolumeDesc desc)
{
    desc.nx = std::max(desc.nx, 1);
    desc.ny = std::max(desc.ny, 1);
    desc.nz = std::max(desc.nz, 1);

    if (data_ != nullptr && volumeDescEquals(desc_, desc)) {
        return;
    }

    destroy();
    desc_ = desc;
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), byteSize()));
}

void DeviceVolume::clear(float value)
{
    if (!isValid()) {
        return;
    }

    const std::size_t count = voxelCount();
    const int block = 256;
    const int grid = static_cast<int>((count + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    fillKernel<<<grid, block>>>(data_, count, value);
    VOL_CUDA_CHECK(cudaGetLastError());
}

void DeviceVolume::destroy()
{
    if (data_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
    }
    desc_ = {};
}

void DeviceVolume::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy device volume cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
