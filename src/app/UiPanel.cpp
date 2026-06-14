#include "app/UiPanel.h"

#include "core/CudaCheck.h"
#include "core/Version.h"

#include <imgui.h>

#include <array>
#include <algorithm>
#include <cstdio>
#include <string>

namespace vollenia {

namespace {

double bytesToMiB(std::size_t bytes)
{
    constexpr double bytes_per_mib = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_mib;
}

VolumePreset indexToPreset(int index)
{
    switch (index) {
    case 0:
        return VolumePreset::Sphere;
    case 1:
        return VolumePreset::Shell;
    case 2:
        return VolumePreset::GaussianBlobs;
    case 3:
        return VolumePreset::LeniaPhantom;
    case 4:
        return VolumePreset::AxisRamp;
    default:
        return VolumePreset::LeniaPhantom;
    }
}

VolumeSource indexToSource(int index)
{
    return index == 0 ? VolumeSource::Synthetic : VolumeSource::Lenia;
}

VolumeRenderMode indexToMode(int index)
{
    switch (index) {
    case 0:
        return VolumeRenderMode::EmissionAbsorption;
    case 1:
        return VolumeRenderMode::MIP;
    case 2:
        return VolumeRenderMode::FirstHit;
    default:
        return VolumeRenderMode::EmissionAbsorption;
    }
}

LeniaSeedPreset indexToLeniaSeedPreset(int index)
{
    switch (index) {
    case 0:
        return LeniaSeedPreset::ReferenceRandomBox;
    case 1:
        return LeniaSeedPreset::CenteredRandomBall;
    case 2:
        return LeniaSeedPreset::AsymmetricGaussianCluster;
    case 3:
        return LeniaSeedPreset::ShellInternalBlobs;
    case 4:
        return LeniaSeedPreset::SmallBlob;
    default:
        return LeniaSeedPreset::ReferenceRandomBox;
    }
}

LeniaParamPreset indexToLeniaParamPreset(int index)
{
    switch (index) {
    case 0:
        return LeniaParamPreset::DiguttomeSaliens;
    case 1:
        return LeniaParamPreset::DiguttomeTardus;
    case 2:
        return LeniaParamPreset::TriguttomeLabens;
    default:
        return LeniaParamPreset::DiguttomeSaliens;
    }
}

KernelCoreType indexToKernelCoreType(int index)
{
    switch (index) {
    case 1:
        return KernelCoreType::ExponentialBump;
    case 2:
        return KernelCoreType::Step;
    case 3:
        return KernelCoreType::Staircase;
    case 0:
    default:
        return KernelCoreType::PolynomialBump;
    }
}

int kernelCoreTypeToIndex(KernelCoreType type)
{
    return std::clamp(static_cast<int>(type) - 1, 0, 3);
}

GrowthFunctionType indexToGrowthFunctionType(int index)
{
    switch (index) {
    case 1:
        return GrowthFunctionType::Gaussian;
    case 2:
        return GrowthFunctionType::Step;
    case 0:
    default:
        return GrowthFunctionType::Polynomial;
    }
}

int growthFunctionTypeToIndex(GrowthFunctionType type)
{
    return std::clamp(static_cast<int>(type) - 1, 0, 2);
}

std::string animalNameOr(const LeniaAnimalCatalog& catalog, int index, const char* fallback)
{
    if (catalog.isLoaded() && index >= 0 && index < catalog.count()) {
        return catalog.animal(index).display_name;
    }
    return fallback;
}

std::string cellsSourceLabel(const LeniaConfig& config, const LeniaAnimalCatalog& catalog)
{
    switch (config.cells_source) {
    case LeniaCellsSource::Procedural:
        return std::string("Procedural: ") + leniaSeedPresetName(config.seed_preset);
    case LeniaCellsSource::Imported:
        return "Animal: " + animalNameOr(catalog, config.cells_source_animal_index, "unknown");
    case LeniaCellsSource::Modified:
        if (config.cells_source_animal_index >= 0) {
            return "Modified from " + animalNameOr(catalog, config.cells_source_animal_index, "animal");
        }
        return "Modified manual";
    default:
        return "Unknown";
    }
}

std::string paramsSourceLabel(const LeniaConfig& config, const LeniaAnimalCatalog& catalog)
{
    switch (config.params_source) {
    case LeniaParamsSource::ParameterPreset:
        return std::string("Debug rule: ") + leniaParamPresetName(config.param_preset);
    case LeniaParamsSource::Animal:
        return "Animal: " + animalNameOr(catalog, config.params_source_animal_index, "unknown");
    case LeniaParamsSource::Modified:
        if (config.params_source_animal_index >= 0) {
            return "Modified from " + animalNameOr(catalog, config.params_source_animal_index, "animal");
        }
        return "Modified manual";
    default:
        return "Unknown";
    }
}

int resolutionToIndex(int resolution, const std::array<int, 6>& resolutions)
{
    for (int i = 0; i < static_cast<int>(resolutions.size()); ++i) {
        if (resolutions[static_cast<std::size_t>(i)] == resolution) {
            return i;
        }
    }
    return 2;
}

} // namespace

UiPanelResult UiPanel::render(
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
    float frame_time_ms)
{
    UiPanelResult result;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 work_pos = viewport != nullptr ? viewport->WorkPos : ImVec2(0.0f, 0.0f);
    const ImVec2 work_size = viewport != nullptr ? viewport->WorkSize : ImVec2(1600.0f, 900.0f);
    constexpr float margin = 16.0f;
    const ImVec2 status_size(360.0f, 300.0f);
    const ImVec2 display_size(360.0f, 360.0f);
    const ImVec2 sim_size(430.0f, 520.0f);
    const ImVec2 catalog_size(430.0f, 410.0f);

    ImGui::SetNextWindowPos(ImVec2(work_pos.x + margin, work_pos.y + margin), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(status_size, ImGuiCond_FirstUseEver);
    ImGui::Begin("VolLenia Status");
    ImGui::TextUnformatted(VOLLENIA_PROJECT_NAME " " VOLLENIA_PROJECT_VERSION);
    ImGui::Separator();
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame time: %.3f ms", frame_time_ms);
    ImGui::Text("OpenGL: %s", gl_version_text != nullptr ? gl_version_text : "Unavailable");
    if (cuda_info.available) {
        ImGui::Text("CUDA device: %s", cuda_info.name.c_str());
        ImGui::Text("CUDA runtime: %s", cudaVersionToString(cuda_info.runtime_version).c_str());
        ImGui::Text("CUDA driver: %s", cudaVersionToString(cuda_info.driver_version).c_str());
        ImGui::Text("Compute capability: %d.%d", cuda_info.compute_major, cuda_info.compute_minor);
        ImGui::Text("Global memory: %.2f GiB", bytesToMiB(cuda_info.global_memory_bytes) / 1024.0);
    } else {
        ImGui::TextUnformatted("CUDA device: unavailable");
        ImGui::TextWrapped("%s", cuda_info.error.c_str());
    }
    ImGui::Separator();
    ImGui::Checkbox("Enable volume render", &render_enabled);
    int source_index = static_cast<int>(volume_source);
    const char* source_names[] = {volumeSourceName(VolumeSource::Synthetic), volumeSourceName(VolumeSource::Lenia)};
    if (ImGui::Combo("Source", &source_index, source_names, IM_ARRAYSIZE(source_names))) {
        volume_source = indexToSource(source_index);
        result.source_changed = true;
    }
    ImGui::Text("Framebuffer: %d x %d", volume_status.framebuffer_width, volume_status.framebuffer_height);
    ImGui::Text("Volume: %d x %d x %d", volume_status.volume_desc.nx, volume_status.volume_desc.ny, volume_status.volume_desc.nz);
    ImGui::Text("Volume bytes: %.2f MiB", bytesToMiB(volume_status.volume_bytes));
    ImGui::TextWrapped("Status: %s", volume_status.status.c_str());
    if (!volume_status.last_error.empty()) {
        ImGui::TextWrapped("Last error: %s", volume_status.last_error.c_str());
    }
    if (ImGui::Button("Quit")) {
        result.quit_requested = true;
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(work_pos.x + work_size.x - display_size.x - margin, work_pos.y + margin), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(display_size, ImGuiCond_FirstUseEver);
    ImGui::Begin("Display & Camera");
    int mode_index = static_cast<int>(render_params.mode);
    const char* mode_names[] = {
        volumeRenderModeName(VolumeRenderMode::EmissionAbsorption),
        volumeRenderModeName(VolumeRenderMode::MIP),
        volumeRenderModeName(VolumeRenderMode::FirstHit),
    };
    if (ImGui::Combo("Render mode", &mode_index, mode_names, IM_ARRAYSIZE(mode_names))) {
        render_params.mode = indexToMode(mode_index);
    }
    ImGui::SliderFloat("Step size", &render_params.step_size, 0.002f, 0.04f, "%.4f");
    ImGui::SliderFloat("Density scale", &render_params.density_scale, 0.1f, 14.0f, "%.2f");
    ImGui::SliderFloat("Threshold", &render_params.threshold, 0.0f, 0.8f, "%.3f");
    ImGui::SliderFloat("Brightness", &render_params.brightness, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Early exit", &render_params.early_exit_transmittance, 0.001f, 0.2f, "%.3f");
    ImGui::SliderInt("Max steps", &render_params.max_steps, 32, 1024);
    ImGui::Separator();
    float distance = camera.settings().distance;
    float fov = camera.settings().fov_y_degrees;
    if (ImGui::SliderFloat("Camera distance", &distance, 0.75f, 12.0f, "%.2f")) {
        camera.setDistance(distance);
    }
    if (ImGui::SliderFloat("Camera FOV Y", &fov, 15.0f, 90.0f, "%.1f deg")) {
        camera.setFovYDegrees(fov);
    }
    if (ImGui::Button("Reset camera")) {
        camera.reset();
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(work_pos.x + margin, work_pos.y + work_size.y - sim_size.y - margin), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(sim_size, ImGuiCond_FirstUseEver);
    ImGui::Begin("Lenia Simulation");
    if (volume_source == VolumeSource::Synthetic) {
        ImGui::TextUnformatted("Synthetic source");
        int preset_index = static_cast<int>(volume_preset);
        const char* preset_names[] = {
            volumePresetName(VolumePreset::Sphere),
            volumePresetName(VolumePreset::Shell),
            volumePresetName(VolumePreset::GaussianBlobs),
            volumePresetName(VolumePreset::LeniaPhantom),
            volumePresetName(VolumePreset::AxisRamp),
        };
        if (ImGui::Combo("Volume preset", &preset_index, preset_names, IM_ARRAYSIZE(preset_names))) {
            volume_preset = indexToPreset(preset_index);
            result.regenerate_volume = true;
        }
        const std::array<int, 6> synthetic_resolutions {64, 96, 128, 160, 192, 256};
        int resolution_index = resolutionToIndex(volume_resolution, synthetic_resolutions);
        const char* resolution_names[] = {"64", "96", "128", "160", "192 exp", "256 exp"};
        if (ImGui::Combo("Volume resolution", &resolution_index, resolution_names, IM_ARRAYSIZE(resolution_names))) {
            volume_resolution = synthetic_resolutions[static_cast<std::size_t>(resolution_index)];
            result.regenerate_volume = true;
        }
        if (ImGui::Button("Regenerate volume")) {
            result.regenerate_volume = true;
        }
    } else {
        ImGui::Checkbox("Play", &lenia_config.playing);
        ImGui::SameLine();
        if (ImGui::Button("Single step")) {
            result.lenia_single_step = true;
        }
        if (ImGui::Button("Reset procedural seed")) {
            result.lenia_reset_seed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Regenerate procedural seed")) {
            result.lenia_regenerate_seed = true;
        }
        ImGui::SliderInt("Steps per frame", &lenia_config.steps_per_frame, 0, 32);
        ImGui::Checkbox("Validate NaN/Inf every step", &lenia_config.validate_nan_inf_every_step);

        const std::array<int, 6> resolutions {64, 96, 128, 160, 192, 256};
        int resolution_index = resolutionToIndex(lenia_config.resolution, resolutions);
        const char* lenia_resolution_names[] = {"64", "96", "128", "160", "192 exp", "256 exp"};
        if (ImGui::Combo("Lenia resolution", &resolution_index, lenia_resolution_names, IM_ARRAYSIZE(lenia_resolution_names))) {
            lenia_config.resolution = resolutions[static_cast<std::size_t>(resolution_index)];
            result.lenia_resolution_changed = true;
        }

        int seed_index = static_cast<int>(lenia_config.seed_preset);
        const char* seed_names[] = {
            leniaSeedPresetName(LeniaSeedPreset::ReferenceRandomBox),
            leniaSeedPresetName(LeniaSeedPreset::CenteredRandomBall),
            leniaSeedPresetName(LeniaSeedPreset::AsymmetricGaussianCluster),
            leniaSeedPresetName(LeniaSeedPreset::ShellInternalBlobs),
            leniaSeedPresetName(LeniaSeedPreset::SmallBlob),
        };
        if (ImGui::Combo("Procedural seed preset", &seed_index, seed_names, IM_ARRAYSIZE(seed_names))) {
            lenia_config.seed_preset = indexToLeniaSeedPreset(seed_index);
            lenia_config.cells_source = LeniaCellsSource::Procedural;
            lenia_config.cells_source_animal_index = -1;
            result.lenia_seed_preset_changed = true;
        }

        int param_index = static_cast<int>(lenia_config.param_preset);
        const char* param_names[] = {
            leniaParamPresetName(LeniaParamPreset::DiguttomeSaliens),
            leniaParamPresetName(LeniaParamPreset::DiguttomeTardus),
            leniaParamPresetName(LeniaParamPreset::TriguttomeLabens),
        };
        if (ImGui::Combo("Debug rule preset", &param_index, param_names, IM_ARRAYSIZE(param_names))) {
            lenia_config.param_preset = indexToLeniaParamPreset(param_index);
            lenia_config.params = leniaParamsForPreset(lenia_config.param_preset);
            lenia_config.params_source = LeniaParamsSource::ParameterPreset;
            lenia_config.params_source_animal_index = -1;
            result.lenia_param_preset_changed = true;
            result.lenia_rebuild_kernel = true;
        }

        if (ImGui::SliderFloat("R", &lenia_config.params.radius, 1.0f, 32.0f, "%.2f")) {
            lenia_config.params_source = LeniaParamsSource::Modified;
            result.lenia_rebuild_kernel = true;
        }
        if (ImGui::SliderFloat("T", &lenia_config.params.T, 1.0f, 60.0f, "%.2f")) {
            lenia_config.params_source = LeniaParamsSource::Modified;
        }
        if (ImGui::SliderFloat("m / mu", &lenia_config.params.mu, 0.0f, 0.5f, "%.4f")) {
            lenia_config.params_source = LeniaParamsSource::Modified;
        }
        if (ImGui::SliderFloat("s / sigma", &lenia_config.params.sigma, 0.001f, 0.08f, "%.4f")) {
            lenia_config.params_source = LeniaParamsSource::Modified;
        }

        int core_index = kernelCoreTypeToIndex(lenia_config.params.kernel_core);
        const char* core_names[] = {
            kernelCoreTypeName(KernelCoreType::PolynomialBump),
            kernelCoreTypeName(KernelCoreType::ExponentialBump),
            kernelCoreTypeName(KernelCoreType::Step),
            kernelCoreTypeName(KernelCoreType::Staircase),
        };
        if (ImGui::Combo("Kernel core", &core_index, core_names, IM_ARRAYSIZE(core_names))) {
            lenia_config.params.kernel_core = indexToKernelCoreType(core_index);
            lenia_config.params_source = LeniaParamsSource::Modified;
            result.lenia_rebuild_kernel = true;
        }

        int growth_index = growthFunctionTypeToIndex(lenia_config.params.growth_function);
        const char* growth_names[] = {
            growthFunctionTypeName(GrowthFunctionType::Polynomial),
            growthFunctionTypeName(GrowthFunctionType::Gaussian),
            growthFunctionTypeName(GrowthFunctionType::Step),
        };
        if (ImGui::Combo("Growth function", &growth_index, growth_names, IM_ARRAYSIZE(growth_names))) {
            lenia_config.params.growth_function = indexToGrowthFunctionType(growth_index);
            lenia_config.params_source = LeniaParamsSource::Modified;
        }

        if (ImGui::SliderInt("Shell count", &lenia_config.params.shell_count, 1, static_cast<int>(lenia_config.params.shell_weights.size()))) {
            lenia_config.params_source = LeniaParamsSource::Modified;
            result.lenia_rebuild_kernel = true;
        }
        for (int i = 0; i < lenia_config.params.shell_count; ++i) {
            char label[16] {};
            std::snprintf(label, sizeof(label), "b%d", i);
            if (ImGui::SliderFloat(label, &lenia_config.params.shell_weights[static_cast<std::size_t>(i)], 0.0f, 1.25f, "%.4f")) {
                lenia_config.params_source = LeniaParamsSource::Modified;
                result.lenia_rebuild_kernel = true;
            }
        }
        if (ImGui::Button("Rebuild kernel")) {
            result.lenia_rebuild_kernel = true;
        }

        ImGui::Separator();
        ImGui::Text("Seed: %u", lenia_config.seed);
        ImGui::Text("Generation: %llu", lenia_status.generation);
        const std::string cells_source = cellsSourceLabel(lenia_config, animal_catalog);
        const std::string params_source = paramsSourceLabel(lenia_config, animal_catalog);
        ImGui::TextWrapped("Cells source: %s", cells_source.c_str());
        ImGui::TextWrapped("Rule source: %s", params_source.c_str());
        ImGui::TextWrapped("Kernel: %s", lenia_status.kernel_status);
        ImGui::TextWrapped("Simulation: %s", lenia_status.simulation_status);
        if (lenia_status.last_error != nullptr && lenia_status.last_error[0] != '\0') {
            ImGui::TextWrapped("Lenia error: %s", lenia_status.last_error);
        }
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(work_pos.x + work_size.x - catalog_size.x - margin, work_pos.y + work_size.y - catalog_size.y - margin), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(catalog_size, ImGuiCond_FirstUseEver);
    ImGui::Begin("Animal Catalog");
    if (!animal_catalog.isLoaded()) {
        ImGui::TextUnformatted("Catalog unavailable");
        ImGui::TextWrapped("%s", animal_catalog.lastError().c_str());
    } else if (animal_catalog.count() <= 0) {
        ImGui::TextUnformatted("Catalog is empty");
    } else {
        lenia_config.selected_animal_index = std::clamp(lenia_config.selected_animal_index, 0, animal_catalog.count() - 1);
        const LeniaAnimalPreset& selected = animal_catalog.animal(lenia_config.selected_animal_index);
        if (ImGui::BeginCombo("Animal preset", selected.display_name.c_str())) {
            for (int i = 0; i < animal_catalog.count(); ++i) {
                const LeniaAnimalPreset& animal = animal_catalog.animal(i);
                const bool is_selected = i == lenia_config.selected_animal_index;
                if (ImGui::Selectable(animal.display_name.c_str(), is_selected)) {
                    lenia_config.selected_animal_index = i;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        const LeniaAnimalPreset& animal = animal_catalog.animal(lenia_config.selected_animal_index);
        ImGui::Text("Code: %s", animal.code.c_str());
        ImGui::Text("Source index: %d", animal.source_index);
        ImGui::Text("Name: %s", animal.display_name.c_str());
        ImGui::Text("Cells dims: %d x %d x %d", animal.cells_desc.nx, animal.cells_desc.ny, animal.cells_desc.nz);
        ImGui::Text("R %.2f  T %.2f  m %.4f  s %.4f", animal.params.radius, animal.params.T, animal.params.mu, animal.params.sigma);
        ImGui::Text("Kernel: %s", kernelCoreTypeName(animal.params.kernel_core));
        ImGui::Text("Growth: %s", growthFunctionTypeName(animal.params.growth_function));
        if (ImGui::Button("Load initial state + rule")) {
            result.lenia_load_animal = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply cells only")) {
            result.lenia_apply_cells_only = true;
        }
        if (ImGui::Button("Apply rule only")) {
            result.lenia_apply_params_only = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset rule to selected animal")) {
            result.lenia_apply_params_only = true;
        }
    }
    ImGui::End();

    return result;
}

} // namespace vollenia
