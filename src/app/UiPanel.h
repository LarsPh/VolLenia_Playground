#pragma once

#include "app/App.h"

namespace vollenia {

class UiPanel {
public:
    bool render(
        const CudaDeviceInfo& cuda_info,
        const char* gl_version_text,
        const CameraSettings& camera,
        PboSmokeUiState& pbo_smoke,
        bool& enable_pbo_smoke_test,
        float fps,
        float frame_time_ms);
};

} // namespace vollenia
