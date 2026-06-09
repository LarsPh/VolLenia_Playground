#pragma once

#include "app/App.h"

namespace vollenia {

class UiPanel {
public:
    bool render(
        const CudaDeviceInfo& cuda_info,
        const char* gl_version_text,
        const CameraSettings& camera,
        float fps,
        float frame_time_ms);
};

} // namespace vollenia
