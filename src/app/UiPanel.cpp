#include "app/UiPanel.h"

#include "core/CudaCheck.h"
#include "core/Version.h"

#include <imgui.h>

#include <array>

namespace vollenia {

namespace {

double bytesToMiB(std::size_t bytes)
{
    constexpr double bytes_per_mib = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_mib;
}

int presetToIndex(VolumePreset preset)
{
    return static_cast<int>(preset);
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

int modeToIndex(VolumeRenderMode mode)
{
    return static_cast<int>(mode);
}

int sourceToIndex(VolumeSource source)
{
    return static_cast<int>(source);
}

VolumeSource indexToSource(int index)
{
    switch (index) {
    case 0:
        return VolumeSource::Synthetic;
    case 1:
        return VolumeSource::Lenia;
    default:
        return VolumeSource::Lenia;
    }
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

} // namespace

UiPanelResult UiPanel::render(
    const CudaDeviceInfo& cuda_info,
    const char* gl_version_text,
    Camera& camera,
    VolumeRenderStatus& volume_status,
    const LeniaStatus& lenia_status,
    VolumeSource& volume_source,
    bool& render_enabled,
    VolumePreset& volume_preset,
    int& volume_resolution,
    LeniaConfig& lenia_config,
    RenderParams& render_params,
    float fps,
    float frame_time_ms)
{
    UiPanelResult result;

    ImGui::Begin("VolLenia Playground");
    ImGui::TextUnformatted(VOLLENIA_PROJECT_NAME " " VOLLENIA_PROJECT_VERSION);
    ImGui::Separator();

    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame time: %.3f ms", frame_time_ms);
    ImGui::Text("OpenGL: %s", gl_version_text != nullptr ? gl_version_text : "Unavailable");

    ImGui::Separator();
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
    ImGui::TextUnformatted("Volume renderer");
    ImGui::Checkbox("Enable volume render", &render_enabled);

    int source_index = sourceToIndex(volume_source);
    const char* source_names[] = {
        volumeSourceName(VolumeSource::Synthetic),
        volumeSourceName(VolumeSource::Lenia),
    };
    if (ImGui::Combo("Source", &source_index, source_names, IM_ARRAYSIZE(source_names))) {
        volume_source = indexToSource(source_index);
        result.source_changed = true;
    }

    const std::array<int, 4> resolutions {64, 96, 128, 160};

    if (volume_source == VolumeSource::Synthetic) {
        ImGui::TextUnformatted("Synthetic source");
        int preset_index = presetToIndex(volume_preset);
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

        int resolution_index = 2;
        for (int i = 0; i < static_cast<int>(resolutions.size()); ++i) {
            if (resolutions[static_cast<std::size_t>(i)] == volume_resolution) {
                resolution_index = i;
                break;
            }
        }
        const char* resolution_names[] = {"64", "96", "128", "160"};
        if (ImGui::Combo("Volume resolution", &resolution_index, resolution_names, IM_ARRAYSIZE(resolution_names))) {
            volume_resolution = resolutions[static_cast<std::size_t>(resolution_index)];
            result.regenerate_volume = true;
        }
        if (ImGui::Button("Regenerate volume")) {
            result.regenerate_volume = true;
        }
    } else {
        ImGui::TextUnformatted("Lenia simulation");
        ImGui::Checkbox("Play", &lenia_config.playing);
        if (ImGui::Button("Single step")) {
            result.lenia_single_step = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset seed")) {
            result.lenia_reset_seed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Regenerate seed")) {
            result.lenia_regenerate_seed = true;
        }
        ImGui::SliderInt("Steps per frame", &lenia_config.steps_per_frame, 0, 32);

        int resolution_index = 2;
        for (int i = 0; i < 3; ++i) {
            if (resolutions[static_cast<std::size_t>(i)] == lenia_config.resolution) {
                resolution_index = i;
                break;
            }
        }
        const char* lenia_resolution_names[] = {"64", "96", "128"};
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
        if (ImGui::Combo("Seed preset", &seed_index, seed_names, IM_ARRAYSIZE(seed_names))) {
            lenia_config.seed_preset = indexToLeniaSeedPreset(seed_index);
            result.lenia_seed_preset_changed = true;
        }

        int param_index = static_cast<int>(lenia_config.param_preset);
        const char* param_names[] = {
            leniaParamPresetName(LeniaParamPreset::DiguttomeSaliens),
            leniaParamPresetName(LeniaParamPreset::DiguttomeTardus),
            leniaParamPresetName(LeniaParamPreset::TriguttomeLabens),
        };
        if (ImGui::Combo("Parameter preset", &param_index, param_names, IM_ARRAYSIZE(param_names))) {
            lenia_config.param_preset = indexToLeniaParamPreset(param_index);
            lenia_config.params = leniaParamsForPreset(lenia_config.param_preset);
            result.lenia_param_preset_changed = true;
            result.lenia_rebuild_kernel = true;
        }

        if (ImGui::SliderFloat("R", &lenia_config.params.radius, 1.0f, 32.0f, "%.2f")) {
            result.lenia_rebuild_kernel = true;
        }
        ImGui::SliderFloat("T", &lenia_config.params.T, 1.0f, 60.0f, "%.2f");
        ImGui::SliderFloat("mu", &lenia_config.params.mu, 0.0f, 0.5f, "%.4f");
        ImGui::SliderFloat("sigma", &lenia_config.params.sigma, 0.001f, 0.08f, "%.4f");
        if (ImGui::Button("Rebuild kernel")) {
            result.lenia_rebuild_kernel = true;
        }

        ImGui::Text("Seed: %u", lenia_config.seed);
        ImGui::Text("Generation: %llu", lenia_status.generation);
        ImGui::TextWrapped("Kernel: %s", lenia_status.kernel_status);
        ImGui::TextWrapped("Simulation: %s", lenia_status.simulation_status);
        ImGui::TextUnformatted("Shell weights:");
        for (int i = 0; i < lenia_config.params.shell_count; ++i) {
            ImGui::SameLine();
            ImGui::Text("%.3f", lenia_config.params.shell_weights[static_cast<std::size_t>(i)]);
        }
        if (lenia_status.last_error != nullptr && lenia_status.last_error[0] != '\0') {
            ImGui::TextWrapped("Lenia error: %s", lenia_status.last_error);
        }
    }

    int mode_index = modeToIndex(render_params.mode);
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
    ImGui::TextUnformatted("Camera");
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

    ImGui::Separator();
    ImGui::Text("Volume: %d x %d x %d", volume_status.volume_desc.nx, volume_status.volume_desc.ny, volume_status.volume_desc.nz);
    ImGui::Text("Volume bytes: %.2f MiB", bytesToMiB(volume_status.volume_bytes));
    ImGui::Text("Framebuffer: %d x %d", volume_status.framebuffer_width, volume_status.framebuffer_height);
    ImGui::Text("Source: %s", volumeSourceName(volume_source));
    if (volume_source == VolumeSource::Synthetic) {
        ImGui::Text("Preset: %s", volumePresetName(volume_status.preset));
    }
    ImGui::Text("Volume: %s", volume_status.volume_valid ? "ready" : "not ready");
    ImGui::Text("Texture: %s", volume_status.texture_valid ? "ready" : "not ready");
    ImGui::TextWrapped("Status: %s", volume_status.status.c_str());
    if (!volume_status.last_error.empty()) {
        ImGui::TextWrapped("Last error: %s", volume_status.last_error.c_str());
    }

    ImGui::Separator();
    if (ImGui::Button("Quit")) {
        result.quit_requested = true;
    }

    ImGui::End();
    return result;
}

} // namespace vollenia
