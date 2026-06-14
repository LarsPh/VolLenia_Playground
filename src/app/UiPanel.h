#pragma once

#include "app/App.h"

namespace vollenia {

struct UiPanelResult {
    bool quit_requested = false;
    bool regenerate_volume = false;
    bool source_changed = false;
    bool lenia_reset_seed = false;
    bool lenia_regenerate_seed = false;
    bool lenia_single_step = false;
    bool lenia_rebuild_kernel = false;
    bool lenia_resolution_changed = false;
    bool lenia_seed_preset_changed = false;
    bool lenia_param_preset_changed = false;
    bool lenia_load_animal = false;
    bool lenia_apply_cells_only = false;
    bool lenia_apply_params_only = false;
};

class UiPanel {
public:
    UiPanelResult render(
        const CudaDeviceInfo& cuda_info,
        const char* gl_version_text,
        Camera& camera,
        VolumeRenderStatus& volume_status,
        const LeniaStatus& lenia_status,
        VolumeSource& volume_source,
        const LeniaAnimalCatalog& animal_catalog,
        bool& render_enabled,
        VolumePreset& volume_preset,
        int& volume_resolution,
        LeniaConfig& lenia_config,
        RenderParams& render_params,
        float fps,
        float frame_time_ms);
};

} // namespace vollenia
