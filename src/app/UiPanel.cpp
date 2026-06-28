#include "app/UiPanel.h"

#include "core/CudaCheck.h"
#include "core/Version.h"

#include <imgui.h>

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
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
    if (index == 0) {
        return VolumeSource::Synthetic;
    }
    if (index == 2) {
        return VolumeSource::ModelSpec;
    }
    return VolumeSource::Lenia;
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

CellResampleMode indexToCellResampleMode(int index)
{
    return index == 0 ? CellResampleMode::Nearest : CellResampleMode::Trilinear;
}

int cellResampleModeToIndex(CellResampleMode mode)
{
    return mode == CellResampleMode::Nearest ? 0 : 1;
}

ModelUpdateMode indexToModelUpdateMode(int index)
{
    return index == 1 ? ModelUpdateMode::Flow : ModelUpdateMode::ExpandedAdditive;
}

int modelUpdateModeToIndex(ModelUpdateMode mode)
{
    return mode == ModelUpdateMode::Flow ? 1 : 0;
}

ChannelRole indexToChannelRole(int index)
{
    switch (index) {
    case 1:
        return ChannelRole::HiddenReserved;
    case 2:
        return ChannelRole::StaticEnv;
    case 3:
        return ChannelRole::RenderOnly;
    case 0:
    default:
        return ChannelRole::Matter;
    }
}

int channelRoleToIndex(ChannelRole role)
{
    switch (role) {
    case ChannelRole::HiddenReserved:
        return 1;
    case ChannelRole::StaticEnv:
        return 2;
    case ChannelRole::RenderOnly:
        return 3;
    case ChannelRole::Matter:
    default:
        return 0;
    }
}

KernelFamily indexToKernelFamily(int index)
{
    return index == 1 ? KernelFamily::SmoothGaussianMixture : KernelFamily::LegacyShell;
}

int kernelFamilyToIndex(KernelFamily family)
{
    return family == KernelFamily::SmoothGaussianMixture ? 1 : 0;
}

GrowthFamily indexToGrowthFamily(int index)
{
    return index == 1 ? GrowthFamily::Gaussian : GrowthFamily::PolynomialLenia3D;
}

int growthFamilyToIndex(GrowthFamily family)
{
    return family == GrowthFamily::Gaussian ? 1 : 0;
}

FlowBorder indexToFlowBorder(int index)
{
    return index == 1 ? FlowBorder::Wall : FlowBorder::Torus;
}

int flowBorderToIndex(FlowBorder border)
{
    return border == FlowBorder::Wall ? 1 : 0;
}

bool inputString(const char* label, std::string& value, std::size_t max_size = 96)
{
    std::array<char, 128> buffer {};
    const std::size_t copy_count = std::min(value.size(), std::min(buffer.size() - 1U, max_size));
    std::memcpy(buffer.data(), value.data(), copy_count);
    if (ImGui::InputText(label, buffer.data(), buffer.size())) {
        value = buffer.data();
        return true;
    }
    return false;
}

void helpMarker(const char* text)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

KernelSpec defaultModelKernel(const ModelSpec& spec)
{
    KernelSpec kernel;
    const int last_channel = std::max(spec.channelCount() - 1, 0);
    kernel.name = "kernel_" + std::to_string(spec.kernelCount());
    kernel.src = last_channel;
    kernel.dst = last_channel;
    kernel.weight = 0.5f;
    kernel.family = KernelFamily::SmoothGaussianMixture;
    kernel.radius = 10.0f;
    kernel.envelope_sharpness = 10.0f;
    kernel.basis = {SmoothKernelBasis {0.25f, 0.08f, 1.0f}};
    kernel.growth = GrowthSpec {GrowthFamily::Gaussian, 0.12f, 0.04f};
    return kernel;
}

void markModelDirty(ModelSpecConfig& config)
{
    config.staged_dirty = true;
    config.edit_error.clear();
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
        return std::string(config.imported_cells_scaled ? "Animal scaled: " : "Animal: ")
            + animalNameOr(catalog, config.cells_source_animal_index, "unknown");
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

template <std::size_t N>
int resolutionToIndex(int resolution, const std::array<int, N>& resolutions)
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
    const ImVec2 catalog_size(460.0f, 500.0f);

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
    const char* source_names[] = {
        volumeSourceName(VolumeSource::Synthetic),
        volumeSourceName(VolumeSource::Lenia),
        volumeSourceName(VolumeSource::ModelSpec),
    };
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
    } else if (volume_source == VolumeSource::Lenia) {
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
    } else {
        ImGui::Checkbox("Play", &modelspec_config.playing);
        ImGui::SameLine();
        if (ImGui::Button("Single step")) {
            result.modelspec_single_step = true;
        }
        if (ImGui::Button("Open model...")) {
            result.modelspec_open_dialog = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Open state...")) {
            result.modelspec_open_state_dialog = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload current")) {
            result.modelspec_reload = true;
        }
        if (!modelspec_config.state_path.empty()) {
            ImGui::SameLine();
            if (ImGui::Button("Reload state")) {
                result.modelspec_reload_state = true;
            }
        }
        if (ImGui::Button("Apply/Rebuild model")) {
            result.modelspec_apply_edits = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset staged edits")) {
            result.modelspec_reset_edits = true;
        }
        if (ImGui::Button("Reset seed")) {
            result.modelspec_reset_seed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Regenerate seed")) {
            result.modelspec_regenerate_seed = true;
        }
        ImGui::SliderInt("Steps per frame", &modelspec_config.steps_per_frame, 0, 32);
        helpMarker("Number of simulation steps advanced for each rendered frame while playing.");
        ImGui::Checkbox("Mass diagnostics", &modelspec_config.mass_diagnostics_enabled);
        helpMarker("When disabled, interactive Lenia Model runs avoid full-volume CPU mass copies. Enable only for debug mass ratios.");

        const std::array<int, 5> resolutions {64, 96, 128, 160, 192};
        int resolution_index = resolutionToIndex(modelspec_config.resolution, resolutions);
        const char* modelspec_resolution_names[] = {"64", "96", "128", "160", "192 exp"};
        if (ImGui::Combo("Model resolution", &resolution_index, modelspec_resolution_names, IM_ARRAYSIZE(modelspec_resolution_names))) {
            modelspec_config.resolution = resolutions[static_cast<std::size_t>(resolution_index)];
            staged_modelspec.default_desc = VolumeDesc {modelspec_config.resolution, modelspec_config.resolution, modelspec_config.resolution};
            result.modelspec_resolution_changed = true;
            markModelDirty(modelspec_config);
        }
        helpMarker("Simulation grid size. 3 channels at 128^3 can fall below 30 FPS on a 4060 Ti; 64 or 96 is the friendlier edit range.");
        const int staged_channel_count = staged_modelspec.channelCount();
        const int max_channel = std::max(staged_channel_count - 1, 0);

        if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
            int render_mode = modelspec_config.composite_render ? 1 : 0;
            const char* model_render_modes[] = {"Single channel", "Composite channels"};
            if (ImGui::Combo("Lenia model render", &render_mode, model_render_modes, IM_ARRAYSIZE(model_render_modes))) {
                modelspec_config.composite_render = render_mode == 1;
            }
            helpMarker("Single channel renders one selected state channel. Composite blends up to the first four enabled channels as colored components.");
            if (modelspec_config.composite_render) {
                ImGui::TextUnformatted("Composite palette: cyan, blue, gold, magenta");
                helpMarker("Composite is a visual overlay of component channels, not a separate cross-channel render physics pass.");
            } else if (staged_channel_count > 0) {
                if (ImGui::SliderInt("Render channel", &modelspec_config.render_channel, 0, max_channel)) {
                    staged_modelspec.render_channel = modelspec_config.render_channel;
                    result.modelspec_render_channel_changed = true;
                    markModelDirty(modelspec_config);
                }
                helpMarker("The state channel sent to the single-channel renderer. Channel indices match the Channels section below.");
            } else {
                ImGui::TextUnformatted("No channel available");
            }
        }

        if (staged_modelspec.channels.empty()) {
            ImGui::Separator();
            ImGui::TextUnformatted("No Lenia model loaded");
        } else {
            if (ImGui::CollapsingHeader("Model globals", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (inputString("Model name", staged_modelspec.name)) {
                    markModelDirty(modelspec_config);
                }
                int update_index = modelUpdateModeToIndex(staged_modelspec.update_mode);
                const char* update_names[] = {"expanded_additive", "flow"};
                if (ImGui::Combo("Update mode", &update_index, update_names, IM_ARRAYSIZE(update_names))) {
                    staged_modelspec.update_mode = indexToModelUpdateMode(update_index);
                    markModelDirty(modelspec_config);
                }
                helpMarker("expanded_additive applies growth directly to matter. flow converts affinity into motion before reintegration.");
                if (ImGui::InputFloat("dt", &staged_modelspec.dt, 0.01f, 0.05f, "%.4f")) {
                    staged_modelspec.dt = std::max(staged_modelspec.dt, 1.0e-5f);
                    markModelDirty(modelspec_config);
                }
                helpMarker("Simulation time step. Larger values move faster but can destabilize dense Flow models.");
                if (ImGui::Checkbox("Hard clip", &staged_modelspec.hard_clip)) {
                    markModelDirty(modelspec_config);
                }
                helpMarker("Clamp channel values to [0, 1] after update. Stage 1 presets usually keep this enabled.");
                if (staged_modelspec.update_mode == ModelUpdateMode::Flow) {
                    if (ImGui::InputFloat("theta_A", &staged_modelspec.flow.theta_A, 0.05f, 0.2f, "%.4f")) {
                        staged_modelspec.flow.theta_A = std::max(staged_modelspec.flow.theta_A, 1.0e-5f);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Matter-density scale used by the Flow alpha term.");
                    if (ImGui::InputFloat("alpha power", &staged_modelspec.flow.alpha_power, 0.05f, 0.2f, "%.4f")) {
                        staged_modelspec.flow.alpha_power = std::max(staged_modelspec.flow.alpha_power, 1.0e-5f);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Exponent controlling how strongly local matter density damps or redirects Flow.");
                    if (ImGui::InputFloat("flow max", &staged_modelspec.flow.flow_max, 0.05f, 0.2f, "%.4f")) {
                        staged_modelspec.flow.flow_max = std::max(staged_modelspec.flow.flow_max, 1.0e-5f);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Maximum transport radius for the gather step. Higher values cost more and move matter farther.");
                    int border_index = flowBorderToIndex(staged_modelspec.flow.border);
                    const char* border_names[] = {"torus", "wall"};
                    if (ImGui::Combo("Border", &border_index, border_names, IM_ARRAYSIZE(border_names))) {
                        staged_modelspec.flow.border = indexToFlowBorder(border_index);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("torus wraps around the volume. wall clamps transport at the boundary.");
                    ImGui::TextUnformatted("Gradient: sobel3d");
                    ImGui::TextUnformatted("Transport sigma: 0.5");
                }
            }

            if (ImGui::CollapsingHeader("Channels", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Channel count: %d", staged_modelspec.channelCount());
                if (ImGui::Button("Add channel")) {
                    ChannelSpec channel;
                    channel.name = "channel_" + std::to_string(staged_modelspec.channelCount());
                    channel.role = ChannelRole::Matter;
                    staged_modelspec.channels.push_back(std::move(channel));
                    markModelDirty(modelspec_config);
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove last channel")) {
                    const int last = staged_modelspec.channelCount() - 1;
                    bool referenced = false;
                    for (const KernelSpec& kernel : staged_modelspec.kernels) {
                        referenced = referenced || kernel.src == last || kernel.dst == last;
                    }
                    if (staged_modelspec.channelCount() <= 1) {
                        modelspec_config.edit_error = "Lenia model must keep at least one channel.";
                    } else if (referenced) {
                        modelspec_config.edit_error = "Cannot remove last channel while a kernel references it.";
                    } else {
                        staged_modelspec.channels.pop_back();
                        staged_modelspec.render_channel = std::clamp(staged_modelspec.render_channel, 0, staged_modelspec.channelCount() - 1);
                        modelspec_config.render_channel = staged_modelspec.render_channel;
                        markModelDirty(modelspec_config);
                    }
                }
                for (int c = 0; c < staged_modelspec.channelCount(); ++c) {
                    ChannelSpec& channel = staged_modelspec.channels[static_cast<std::size_t>(c)];
                    char label[64] {};
                    std::snprintf(label, sizeof(label), "Channel %d: %s", c, channel.name.c_str());
                    if (ImGui::TreeNode(label)) {
                        char name_label[32] {};
                        std::snprintf(name_label, sizeof(name_label), "Name##channel%d", c);
                        if (inputString(name_label, channel.name)) {
                            markModelDirty(modelspec_config);
                        }
                        int role_index = channelRoleToIndex(channel.role);
                        const char* role_names[] = {"matter", "hidden_reserved", "static_env", "render_only"};
                        char role_label[32] {};
                        std::snprintf(role_label, sizeof(role_label), "Role##channel%d", c);
                        if (ImGui::Combo(role_label, &role_index, role_names, IM_ARRAYSIZE(role_names))) {
                            channel.role = indexToChannelRole(role_index);
                            markModelDirty(modelspec_config);
                        }
                        helpMarker("Stage 1 updates and transports matter channels. Other roles are parsed and visible but not dynamically updated yet.");
                        if (c < kMaxCompositeChannels) {
                            char enabled_label[48] {};
                            std::snprintf(enabled_label, sizeof(enabled_label), "Composite visible##channel%d", c);
                            ImGui::Checkbox(enabled_label, &modelspec_config.composite_enabled[static_cast<std::size_t>(c)]);
                            helpMarker("Enable this channel in composite render mode.");
                            char intensity_label[48] {};
                            std::snprintf(intensity_label, sizeof(intensity_label), "Composite intensity##channel%d", c);
                            ImGui::SliderFloat(intensity_label, &modelspec_config.composite_intensity[static_cast<std::size_t>(c)], 0.0f, 3.0f, "%.2f");
                            helpMarker("Visual brightness multiplier for this channel in composite render mode.");
                        }
                        ImGui::TreePop();
                    }
                }
            }

            if (ImGui::CollapsingHeader("Kernels", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Kernel count: %d", staged_modelspec.kernelCount());
                if (ImGui::Button("Add kernel")) {
                    staged_modelspec.kernels.push_back(defaultModelKernel(staged_modelspec));
                    modelspec_config.selected_kernel = staged_modelspec.kernelCount() - 1;
                    markModelDirty(modelspec_config);
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove selected kernel")) {
                    if (staged_modelspec.kernelCount() <= 1) {
                        modelspec_config.edit_error = "Lenia model must keep at least one kernel.";
                    } else {
                        const int selected = std::clamp(modelspec_config.selected_kernel, 0, staged_modelspec.kernelCount() - 1);
                        staged_modelspec.kernels.erase(staged_modelspec.kernels.begin() + selected);
                        modelspec_config.selected_kernel = std::clamp(selected, 0, staged_modelspec.kernelCount() - 1);
                        markModelDirty(modelspec_config);
                    }
                }
                modelspec_config.selected_kernel = std::clamp(modelspec_config.selected_kernel, 0, std::max(staged_modelspec.kernelCount() - 1, 0));
                if (staged_modelspec.kernelCount() > 0) {
                    std::string preview = staged_modelspec.kernels[static_cast<std::size_t>(modelspec_config.selected_kernel)].name;
                    if (ImGui::BeginCombo("Selected kernel", preview.c_str())) {
                        for (int k = 0; k < staged_modelspec.kernelCount(); ++k) {
                            const KernelSpec& kernel = staged_modelspec.kernels[static_cast<std::size_t>(k)];
                            char item_label[128] {};
                            std::snprintf(item_label, sizeof(item_label), "%d: %s (%d -> %d)", k, kernel.name.c_str(), kernel.src, kernel.dst);
                            const bool selected = k == modelspec_config.selected_kernel;
                            if (ImGui::Selectable(item_label, selected)) {
                                modelspec_config.selected_kernel = k;
                            }
                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    KernelSpec& kernel = staged_modelspec.kernels[static_cast<std::size_t>(modelspec_config.selected_kernel)];
                    if (inputString("Kernel name", kernel.name)) {
                        markModelDirty(modelspec_config);
                    }
                    if (ImGui::SliderInt("src", &kernel.src, 0, max_channel)) {
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Source channel sampled by this kernel.");
                    if (ImGui::SliderInt("dst", &kernel.dst, 0, max_channel)) {
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Destination channel receiving this kernel's growth contribution.");
                    ImGui::Text("Graph: %d -> %d", kernel.src, kernel.dst);
                    if (ImGui::InputFloat("weight", &kernel.weight, 0.01f, 0.1f, "%.4f")) {
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Signed contribution scale. Negative weights act as inhibitory interactions.");
                    int family_index = kernelFamilyToIndex(kernel.family);
                    const char* family_names[] = {"legacy_shell", "smooth_gaussian_mixture"};
                    if (ImGui::Combo("Kernel family", &family_index, family_names, IM_ARRAYSIZE(family_names))) {
                        kernel.family = indexToKernelFamily(family_index);
                        if (kernel.family == KernelFamily::SmoothGaussianMixture && kernel.basis.empty()) {
                            kernel.basis.push_back(SmoothKernelBasis {});
                        }
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("smooth_gaussian_mixture is the preferred Stage 1 family; legacy_shell is for compatibility/debug conversion.");
                    if (ImGui::InputFloat("R", &kernel.radius, 0.25f, 1.0f, "%.3f")) {
                        kernel.radius = std::max(kernel.radius, 1.0e-5f);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Kernel radius in voxels. Larger R samples wider neighborhoods and costs the same FFT size but changes dynamics strongly.");
                    if (ImGui::InputFloat("envelope sharpness", &kernel.envelope_sharpness, 0.25f, 1.0f, "%.3f")) {
                        kernel.envelope_sharpness = std::max(kernel.envelope_sharpness, 1.0e-5f);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Controls how sharply the smooth kernel falls off near its radius.");

                    int growth_index = growthFamilyToIndex(kernel.growth.family);
                    const char* growth_names[] = {"polynomial_lenia3d", "gaussian"};
                    if (ImGui::Combo("Growth family", &growth_index, growth_names, IM_ARRAYSIZE(growth_names))) {
                        kernel.growth.family = indexToGrowthFamily(growth_index);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Growth curve applied to the kernel potential before accumulation.");
                    if (ImGui::InputFloat("growth mu", &kernel.growth.mu, 0.005f, 0.02f, "%.5f")) {
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Preferred potential value for this growth curve.");
                    if (ImGui::InputFloat("growth sigma", &kernel.growth.sigma, 0.001f, 0.01f, "%.5f")) {
                        kernel.growth.sigma = std::max(kernel.growth.sigma, 1.0e-6f);
                        markModelDirty(modelspec_config);
                    }
                    helpMarker("Growth curve width. Smaller values make the response more selective.");

                    if (kernel.family == KernelFamily::SmoothGaussianMixture) {
                        ImGui::Separator();
                        ImGui::Text("Basis count: %d", static_cast<int>(kernel.basis.size()));
                        if (ImGui::Button("Add basis")) {
                            kernel.basis.push_back(SmoothKernelBasis {});
                            markModelDirty(modelspec_config);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Remove last basis")) {
                            if (kernel.basis.size() <= 1U) {
                                modelspec_config.edit_error = "Smooth kernels must keep at least one basis.";
                            } else {
                                kernel.basis.pop_back();
                                markModelDirty(modelspec_config);
                            }
                        }
                        for (int b = 0; b < static_cast<int>(kernel.basis.size()); ++b) {
                            SmoothKernelBasis& basis = kernel.basis[static_cast<std::size_t>(b)];
                            char basis_label[32] {};
                            std::snprintf(basis_label, sizeof(basis_label), "Basis %d", b);
                            if (ImGui::TreeNode(basis_label)) {
                                char center_label[32] {};
                                std::snprintf(center_label, sizeof(center_label), "center##basis%d", b);
                                if (ImGui::SliderFloat(center_label, &basis.center, 0.0f, 1.5f, "%.4f")) {
                                    markModelDirty(modelspec_config);
                                }
                                helpMarker("Radial location of this smooth kernel lobe, normalized by R.");
                                char width_label[32] {};
                                std::snprintf(width_label, sizeof(width_label), "width##basis%d", b);
                                if (ImGui::SliderFloat(width_label, &basis.width, 0.001f, 0.5f, "%.4f")) {
                                    markModelDirty(modelspec_config);
                                }
                                helpMarker("Width of this radial lobe.");
                                char amp_label[32] {};
                                std::snprintf(amp_label, sizeof(amp_label), "amplitude##basis%d", b);
                                if (ImGui::SliderFloat(amp_label, &basis.amplitude, -2.0f, 2.0f, "%.4f")) {
                                    markModelDirty(modelspec_config);
                                }
                                helpMarker("Signed amplitude of this radial lobe before kernel normalization.");
                                ImGui::TreePop();
                            }
                        }
                    } else {
                        ImGui::Separator();
                        if (kernel.shell_weights.empty()) {
                            ImGui::TextUnformatted("No legacy shell weights in this kernel.");
                        }
                        for (int i = 0; i < static_cast<int>(kernel.shell_weights.size()); ++i) {
                            char label[32] {};
                            std::snprintf(label, sizeof(label), "shell b%d", i);
                            if (ImGui::SliderFloat(label, &kernel.shell_weights[static_cast<std::size_t>(i)], 0.0f, 2.0f, "%.4f")) {
                                markModelDirty(modelspec_config);
                            }
                        }
                    }
                }
            }
        }

        ImGui::Separator();
        ImGui::TextWrapped("Path: %s", modelspec_config.model_path.c_str());
        ImGui::TextWrapped("Model: %s", modelspec_status.model_name.c_str());
        ImGui::Text("Mode: %s", modelspec_status.update_mode);
        ImGui::Text("Channels: %d", modelspec_status.channel_count);
        ImGui::Text("Kernels: %d", modelspec_status.kernel_count);
        ImGui::Text("Render channel: %d", modelspec_status.render_channel);
        ImGui::Text("Generation: %llu", modelspec_status.generation);
        ImGui::Text("Transport dd: %d", modelspec_status.reintegration_dd);
        if (modelspec_status.mass_diagnostics_enabled) {
            ImGui::Text("Mass: %.6f -> %.6f", modelspec_status.mass_before, modelspec_status.mass_after);
            ImGui::Text("Mass ratio: %.6f", modelspec_status.mass_ratio);
        } else {
            ImGui::TextUnformatted("Mass: diagnostics disabled");
        }
        if (!modelspec_config.state_path.empty()) {
            ImGui::TextWrapped("State: %s", modelspec_config.state_path.c_str());
        }
        ImGui::TextWrapped("Simulation: %s", modelspec_status.status.c_str());
        if (modelspec_config.staged_dirty) {
            ImGui::TextUnformatted("Staged edits pending Apply/Rebuild model");
        }
        if (!modelspec_config.edit_error.empty()) {
            ImGui::TextWrapped("Lenia model edit error: %s", modelspec_config.edit_error.c_str());
        }
        if (!modelspec_config.load_error.empty()) {
            ImGui::TextWrapped("Lenia model load error: %s", modelspec_config.load_error.c_str());
        }
        if (!modelspec_status.last_error.empty()) {
            ImGui::TextWrapped("Lenia model error: %s", modelspec_status.last_error.c_str());
        }
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(work_pos.x + work_size.x - catalog_size.x - margin, work_pos.y + work_size.y - catalog_size.y - margin), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(catalog_size, ImGuiCond_FirstUseEver);
    ImGui::Begin("Animal Catalog");
    if (ImGui::Button("Open catalog...")) {
        result.lenia_open_catalog_dialog = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reload current")) {
        result.lenia_reload_catalog = true;
    }
    if (!animal_catalog_error.empty()) {
        ImGui::TextWrapped("Catalog error: %s", animal_catalog_error.c_str());
    }
    ImGui::Separator();
    if (!animal_catalog.isLoaded()) {
        ImGui::TextUnformatted("Catalog unavailable");
        ImGui::TextWrapped("Path: %s", lenia_config.animal_catalog_path.c_str());
        ImGui::TextWrapped("%s", animal_catalog.lastError().c_str());
    } else if (animal_catalog.count() <= 0) {
        ImGui::TextUnformatted("Catalog is empty");
        ImGui::TextWrapped("Path: %s", lenia_config.animal_catalog_path.c_str());
    } else {
        ImGui::TextWrapped("Path: %s", animal_catalog.manifestPath().string().c_str());
        ImGui::Separator();
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
        if (!animal.cname.empty()) {
            ImGui::TextWrapped("CName: %s", animal.cname.c_str());
        }
        ImGui::Text("Cells dims: %d x %d x %d", animal.cells_desc.nx, animal.cells_desc.ny, animal.cells_desc.nz);
        ImGui::Text(
            "Simulation dims: %d x %d x %d",
            animal.simulation_desc.nx,
            animal.simulation_desc.ny,
            animal.simulation_desc.nz);
        ImGui::Text("Resolution policy: %s", animal.resolution_policy.c_str());
        ImGui::Text("R %.2f  T %.2f  m %.4f  s %.4f", animal.params.radius, animal.params.T, animal.params.mu, animal.params.sigma);
        ImGui::Text("Kernel: %s", kernelCoreTypeName(animal.params.kernel_core));
        ImGui::Text("Growth: %s", growthFunctionTypeName(animal.params.growth_function));
        ImGui::Separator();
        ImGui::SliderFloat("Imported cells scale", &lenia_config.imported_cell_scale, 1.0f, 8.0f, "%.2f");
        int resample_mode_index = cellResampleModeToIndex(lenia_config.cell_resample_mode);
        const char* resample_mode_names[] = {
            cellResampleModeName(CellResampleMode::Nearest),
            cellResampleModeName(CellResampleMode::Trilinear),
        };
        if (ImGui::Combo("Resample mode", &resample_mode_index, resample_mode_names, IM_ARRAYSIZE(resample_mode_names))) {
            lenia_config.cell_resample_mode = indexToCellResampleMode(resample_mode_index);
        }
        ImGui::Checkbox("Auto-scale R with cells", &lenia_config.auto_scale_imported_R);
        ImGui::Text(
            "Scaled dims: %d x %d x %d",
            std::max(1, static_cast<int>(animal.cells_desc.nx * lenia_config.imported_cell_scale + 0.5f)),
            std::max(1, static_cast<int>(animal.cells_desc.ny * lenia_config.imported_cell_scale + 0.5f)),
            std::max(1, static_cast<int>(animal.cells_desc.nz * lenia_config.imported_cell_scale + 0.5f)));
        if (ImGui::Button("Load native state + native rule")) {
            result.lenia_load_animal = true;
        }
        if (ImGui::Button("Load scaled state + scaled rule")) {
            result.lenia_load_scaled_animal = true;
        }
        if (ImGui::Button("Apply cells only")) {
            result.lenia_apply_cells_only = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply scaled cells only")) {
            result.lenia_apply_scaled_cells_only = true;
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
