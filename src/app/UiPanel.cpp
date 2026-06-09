#include "app/UiPanel.h"

#include "core/CudaCheck.h"
#include "core/Version.h"

#include <imgui.h>

namespace vollenia {

namespace {

double bytesToGiB(std::size_t bytes)
{
    constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_gib;
}

} // namespace

bool UiPanel::render(
    const CudaDeviceInfo& cuda_info,
    const char* gl_version_text,
    const CameraSettings& camera,
    float fps,
    float frame_time_ms)
{
    bool quit_requested = false;

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
        ImGui::Text("Global memory: %.2f GiB", bytesToGiB(cuda_info.global_memory_bytes));
    } else {
        ImGui::TextUnformatted("CUDA device: unavailable");
        ImGui::TextWrapped("%s", cuda_info.error.c_str());
    }

    ImGui::Separator();
    ImGui::Text("Camera distance: %.2f", camera.distance);
    ImGui::Text("Camera FOV Y: %.1f deg", camera.fov_y_degrees);

    ImGui::Separator();
    if (ImGui::Button("Quit")) {
        quit_requested = true;
    }

    ImGui::End();
    return quit_requested;
}

} // namespace vollenia
