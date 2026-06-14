#pragma once

#include "core/VolumeDesc.h"

#include <cstddef>

namespace vollenia {

class DeviceVolume {
public:
    DeviceVolume() = default;
    ~DeviceVolume() noexcept;

    DeviceVolume(const DeviceVolume&) = delete;
    DeviceVolume& operator=(const DeviceVolume&) = delete;

    void resize(VolumeDesc desc);
    void clear(float value = 0.0f);
    void destroy();

    [[nodiscard]] float* data() { return data_; }
    [[nodiscard]] const float* data() const { return data_; }
    [[nodiscard]] DeviceVolumeView view() const { return DeviceVolumeView {desc_, data_}; }
    [[nodiscard]] const VolumeDesc& desc() const { return desc_; }
    [[nodiscard]] std::size_t voxelCount() const { return volumeVoxelCount(desc_); }
    [[nodiscard]] std::size_t byteSize() const { return volumeByteSize(desc_); }
    [[nodiscard]] bool isValid() const { return data_ != nullptr && isValidVolumeDesc(desc_); }

private:
    void destroyNoThrow() noexcept;

    VolumeDesc desc_;
    float* data_ = nullptr;
};

} // namespace vollenia
