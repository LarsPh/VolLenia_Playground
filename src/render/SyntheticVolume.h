#pragma once

#include "render/RenderParams.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace vollenia {

class SyntheticVolume {
public:
    SyntheticVolume() = default;
    ~SyntheticVolume() noexcept;

    SyntheticVolume(const SyntheticVolume&) = delete;
    SyntheticVolume& operator=(const SyntheticVolume&) = delete;

    void resize(VolumeDesc desc);
    void generate(VolumePreset preset, unsigned int seed);
    void destroy();

    [[nodiscard]] const VolumeDesc& desc() const { return desc_; }
    [[nodiscard]] const float* deviceData() const { return data_; }
    [[nodiscard]] float* deviceData() { return data_; }
    [[nodiscard]] std::size_t voxelCount() const;
    [[nodiscard]] std::size_t byteSize() const { return voxelCount() * sizeof(float); }
    [[nodiscard]] bool isValid() const { return data_ != nullptr && desc_.nx > 0 && desc_.ny > 0 && desc_.nz > 0; }
    [[nodiscard]] VolumePreset preset() const { return preset_; }
    [[nodiscard]] unsigned int seed() const { return seed_; }

private:
    void destroyNoThrow() noexcept;

    VolumeDesc desc_;
    float* data_ = nullptr;
    VolumePreset preset_ = VolumePreset::Sphere;
    unsigned int seed_ = 1;
};

} // namespace vollenia
