#pragma once

#include "core/VolumeDesc.h"
#include "render/CudaPbo.h"
#include "render/RenderParams.h"

#include <cuda_runtime.h>

#include <array>

namespace vollenia {

constexpr int kMaxCompositeChannels = 4;

struct CompositeChannelRenderParams {
    bool enabled = true;
    float intensity = 1.0f;
    float3 color {0.2f, 0.88f, 0.78f};
};

class CudaVolumeRenderer {
public:
    CudaVolumeRenderer() = default;
    ~CudaVolumeRenderer() noexcept;

    CudaVolumeRenderer(const CudaVolumeRenderer&) = delete;
    CudaVolumeRenderer& operator=(const CudaVolumeRenderer&) = delete;

    void uploadVolume(DeviceVolumeView volume);
    void uploadCompositeVolumes(const std::array<DeviceVolumeView, kMaxCompositeChannels>& volumes, int channel_count);
    void render(PboMapping output, int width, int height, const CameraFrame& camera, const RenderParams& params);
    void renderComposite(
        PboMapping output,
        int width,
        int height,
        const CameraFrame& camera,
        const RenderParams& params,
        const std::array<CompositeChannelRenderParams, kMaxCompositeChannels>& channels,
        int channel_count);
    void destroy();

    [[nodiscard]] bool hasTexture() const { return texture_ != 0 && array_ != nullptr; }
    [[nodiscard]] bool hasCompositeTextures() const { return composite_channel_count_ > 0; }
    [[nodiscard]] const VolumeDesc& volumeDesc() const { return volume_desc_; }

private:
    void createTextureStorage(VolumeDesc desc);
    void createCompositeTextureStorage(VolumeDesc desc, int channel_count);
    void destroyNoThrow() noexcept;

    cudaArray_t array_ = nullptr;
    cudaTextureObject_t texture_ = 0;
    std::array<cudaArray_t, kMaxCompositeChannels> composite_arrays_ {};
    std::array<cudaTextureObject_t, kMaxCompositeChannels> composite_textures_ {};
    VolumeDesc volume_desc_;
    int composite_channel_count_ = 0;
};

} // namespace vollenia
