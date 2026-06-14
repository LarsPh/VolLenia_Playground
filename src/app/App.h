#pragma once

#include "app/Camera.h"
#include "render/RenderParams.h"
#include "sim/LeniaParams.h"

#include <cstddef>
#include <memory>
#include <string>

struct GLFWwindow;

namespace vollenia {

class CudaPbo;
class CudaVolumeRenderer;
class GlDisplay;
class LeniaSimulation;
class SyntheticVolume;

struct WindowConfig {
    int width = 1280;
    int height = 720;
    std::string title = "VolLenia Playground";
};

struct AppConfig {
    WindowConfig window;
    CameraSettings camera;
    RenderParams render;
    VolumeDesc volume;
    VolumePreset preset = VolumePreset::LeniaPhantom;
    VolumeSource source = VolumeSource::Lenia;
};

struct LeniaConfig {
    bool playing = true;
    int steps_per_frame = 1;
    int resolution = 128;
    unsigned int seed = 1;
    LeniaSeedPreset seed_preset = LeniaSeedPreset::ReferenceRandomBox;
    LeniaParamPreset param_preset = LeniaParamPreset::DiguttomeSaliens;
    LeniaParams params = leniaParamsForPreset(LeniaParamPreset::DiguttomeSaliens);
};

struct CudaDeviceInfo {
    bool available = false;
    int device_index = 0;
    std::string name;
    int runtime_version = 0;
    int driver_version = 0;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t global_memory_bytes = 0;
    std::string error;
};

class App {
public:
    App();
    ~App();

    App(const App&) = delete; // disable copy constructor since it holds GLFW window and OpenGL context (possible double free)
    App& operator=(const App&) = delete; // disable copy assignment

    int run();

private:
    void initialize();
    void mainLoop();
    void shutdown() noexcept; // called by destructor
    void destroyRenderResources() noexcept;
    void updateFramebufferResources();
    void handleCameraInput();
    void renderVolumeFrame();
    void renderSyntheticVolumeFrame();
    void renderLeniaVolumeFrame();
    void drawUploadedVolume(DeviceVolumeView volume, const char* status_text);

    [[nodiscard]] AppConfig loadConfig(const std::string& path) const; // nodiscard since return something important
    void loadLeniaConfig(const std::string& path);
    [[nodiscard]] CudaDeviceInfo queryCudaDeviceInfo() const;

    AppConfig config_;
    Camera camera_;
    CudaDeviceInfo cuda_info_;
    std::unique_ptr<CudaPbo> pbo_;
    std::unique_ptr<CudaVolumeRenderer> volume_renderer_;
    std::unique_ptr<GlDisplay> display_;
    std::unique_ptr<SyntheticVolume> synthetic_volume_;
    std::unique_ptr<LeniaSimulation> lenia_simulation_;
    GLFWwindow* window_ = nullptr;
    const char* gl_version_text_ = "Unavailable";
    RenderParams render_params_;
    VolumeRenderStatus volume_status_;
    LeniaConfig lenia_config_;
    LeniaStatus lenia_status_;
    VolumeSource volume_source_ = VolumeSource::Lenia;
    VolumePreset volume_preset_ = VolumePreset::LeniaPhantom;
    int volume_resolution_ = 128;
    unsigned int volume_seed_ = 1;
    bool render_enabled_ = true;
    bool volume_dirty_ = true;
    bool renderer_volume_dirty_ = true;
    bool lenia_sim_dirty_ = true;
    bool lenia_seed_dirty_ = true;
    bool lenia_kernel_dirty_ = true;
    bool lenia_single_step_requested_ = false;
    int framebuffer_width_ = 0;
    int framebuffer_height_ = 0;
    float animation_time_seconds_ = 0.0f;
    bool imgui_initialized_ = false;
    bool glfw_initialized_ = false;
};

} // namespace vollenia
