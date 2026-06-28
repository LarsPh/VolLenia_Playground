#pragma once

#include "core/VolumeDesc.h"

#include <cstddef>

namespace vollenia {

class DeviceMultiVolume {
public:
    DeviceMultiVolume() = default;
    ~DeviceMultiVolume() noexcept;

    DeviceMultiVolume(const DeviceMultiVolume&) = delete;
    DeviceMultiVolume& operator=(const DeviceMultiVolume&) = delete;
    DeviceMultiVolume(DeviceMultiVolume&& other) noexcept;
    DeviceMultiVolume& operator=(DeviceMultiVolume&& other) noexcept;

    void resize(VolumeDesc desc, int channels);
    void clear(float value = 0.0f);
    void destroy();
    void swap(DeviceMultiVolume& other) noexcept;

    [[nodiscard]] float* data() { return data_; }
    [[nodiscard]] const float* data() const { return data_; }
    [[nodiscard]] float* channelData(int channel);
    [[nodiscard]] const float* channelData(int channel) const;
    [[nodiscard]] DeviceVolumeView channelView(int channel) const;
    [[nodiscard]] const VolumeDesc& desc() const { return desc_; }
    [[nodiscard]] int channelCount() const { return channels_; }
    [[nodiscard]] std::size_t voxelCount() const { return volumeVoxelCount(desc_); }
    [[nodiscard]] std::size_t valueCount() const { return voxelCount() * static_cast<std::size_t>(channels_); }
    [[nodiscard]] std::size_t byteSize() const { return valueCount() * sizeof(float); }
    [[nodiscard]] bool isValid() const { return data_ != nullptr && isValidVolumeDesc(desc_) && channels_ > 0; }

private:
    void destroyNoThrow() noexcept;

    VolumeDesc desc_;
    int channels_ = 0;
    float* data_ = nullptr;
};

} // namespace vollenia
