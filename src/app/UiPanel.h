#pragma once

#include "app/App.h"

namespace vollenia {

struct UiPanelResult {
    bool quit_requested = false;
    bool regenerate_volume = false;
};

class UiPanel {
public:
    UiPanelResult render(
        const CudaDeviceInfo& cuda_info,
        const char* gl_version_text,
        Camera& camera,
        VolumeRenderStatus& volume_status,
        bool& render_enabled,
        VolumePreset& volume_preset,
        int& volume_resolution,
        RenderParams& render_params,
        float fps,
        float frame_time_ms);
};

} // namespace vollenia
