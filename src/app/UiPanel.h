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
    bool lenia_load_scaled_animal = false;
    bool lenia_apply_cells_only = false;
    bool lenia_apply_scaled_cells_only = false;
    bool lenia_apply_params_only = false;
    bool lenia_reload_catalog = false;
    bool lenia_open_catalog_dialog = false;
    bool modelspec_open_dialog = false;
    bool modelspec_reload = false;
    bool modelspec_reset_seed = false;
    bool modelspec_regenerate_seed = false;
    bool modelspec_single_step = false;
    bool modelspec_resolution_changed = false;
    bool modelspec_render_channel_changed = false;
    bool modelspec_apply_edits = false;
    bool modelspec_reset_edits = false;
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
        const std::string& animal_catalog_error,
        ModelSpecConfig& modelspec_config,
        const ExpandedFlowStatus& modelspec_status,
        ModelSpec& staged_modelspec,
        bool& render_enabled,
        VolumePreset& volume_preset,
        int& volume_resolution,
        LeniaConfig& lenia_config,
        RenderParams& render_params,
        float fps,
        float frame_time_ms);
};

} // namespace vollenia
