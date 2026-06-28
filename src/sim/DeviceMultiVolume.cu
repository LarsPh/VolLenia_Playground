#include "sim/DeviceMultiVolume.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>

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

DeviceMultiVolume::~DeviceMultiVolume() noexcept
{
    destroyNoThrow();
}

DeviceMultiVolume::DeviceMultiVolume(DeviceMultiVolume&& other) noexcept
{
    swap(other);
}

DeviceMultiVolume& DeviceMultiVolume::operator=(DeviceMultiVolume&& other) noexcept
{
    if (this != &other) {
        destroyNoThrow();
        swap(other);
    }
    return *this;
}

void DeviceMultiVolume::swap(DeviceMultiVolume& other) noexcept
{
    std::swap(desc_, other.desc_);
    std::swap(channels_, other.channels_);
    std::swap(data_, other.data_);
}

void DeviceMultiVolume::resize(VolumeDesc desc, int channels)
{
    desc.nx = std::max(desc.nx, 1);
    desc.ny = std::max(desc.ny, 1);
    desc.nz = std::max(desc.nz, 1);
    channels = std::max(channels, 1);

    if (data_ != nullptr && volumeDescEquals(desc_, desc) && channels_ == channels) {
        return;
    }

    destroy();
    desc_ = desc;
    channels_ = channels;
    VOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), byteSize()));
}

void DeviceMultiVolume::clear(float value)
{
    if (!isValid()) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((valueCount() + static_cast<std::size_t>(block) - 1U) / static_cast<std::size_t>(block));
    fillKernel<<<grid, block>>>(data_, valueCount(), value);
    VOL_CUDA_CHECK(cudaGetLastError());
}

float* DeviceMultiVolume::channelData(int channel)
{
    if (!isValid() || channel < 0 || channel >= channels_) {
        throw std::out_of_range("DeviceMultiVolume channel index out of range");
    }
    return data_ + static_cast<std::size_t>(channel) * voxelCount();
}

const float* DeviceMultiVolume::channelData(int channel) const
{
    if (!isValid() || channel < 0 || channel >= channels_) {
        throw std::out_of_range("DeviceMultiVolume channel index out of range");
    }
    return data_ + static_cast<std::size_t>(channel) * voxelCount();
}

DeviceVolumeView DeviceMultiVolume::channelView(int channel) const
{
    return DeviceVolumeView {desc_, channelData(channel)};
}

void DeviceMultiVolume::destroy()
{
    if (data_ != nullptr) {
        VOL_CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
    }
    desc_ = {};
    channels_ = 0;
}

void DeviceMultiVolume::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy multi-channel device volume cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
