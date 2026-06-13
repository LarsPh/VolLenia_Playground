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

} // namespace

UiPanelResult UiPanel::render(
    const CudaDeviceInfo& cuda_info,
    const char* gl_version_text,
    Camera& camera,
    VolumeRenderStatus& volume_status,
    bool& render_enabled,
    VolumePreset& volume_preset,
    int& volume_resolution,
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

    const std::array<int, 4> resolutions {64, 96, 128, 160};
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
    ImGui::Text("Preset: %s", volumePresetName(volume_status.preset));
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
