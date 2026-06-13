#pragma once

#include "render/CudaPbo.h"
#include "render/RenderParams.h"
#include "render/SyntheticVolume.h"

#include <cuda_runtime.h>

namespace vollenia {

class CudaVolumeRenderer {
public:
    CudaVolumeRenderer() = default;
    ~CudaVolumeRenderer() noexcept;

    CudaVolumeRenderer(const CudaVolumeRenderer&) = delete;
    CudaVolumeRenderer& operator=(const CudaVolumeRenderer&) = delete;

    void setVolume(const SyntheticVolume& volume);
    void render(PboMapping output, int width, int height, const CameraFrame& camera, const RenderParams& params);
    void destroy();

    [[nodiscard]] bool hasTexture() const { return texture_ != 0 && array_ != nullptr; }
    [[nodiscard]] const VolumeDesc& volumeDesc() const { return volume_desc_; }

private:
    void destroyNoThrow() noexcept;

    cudaArray_t array_ = nullptr;
    cudaTextureObject_t texture_ = 0;
    VolumeDesc volume_desc_;
};

} // namespace vollenia
