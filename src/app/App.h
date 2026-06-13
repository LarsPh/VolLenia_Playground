#pragma once

#include "app/Camera.h"

#include <cstddef>
#include <memory>
#include <string>

struct GLFWwindow;

namespace vollenia {

class CudaPbo;
class GlDisplay;

struct WindowConfig {
    int width = 1280;
    int height = 720;
    std::string title = "VolLenia Playground";
};

struct AppConfig {
    WindowConfig window;
    CameraSettings camera;
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

struct PboSmokeUiState {
    bool resource_valid = false;
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    std::size_t pbo_byte_size = 0;
    float animation_time_seconds = 0.0f;
    std::string status = "Not initialized";
    std::string last_error;
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
    void renderPboSmokeFrame();

    [[nodiscard]] AppConfig loadConfig(const std::string& path) const; // nodiscard since return something important
    [[nodiscard]] CudaDeviceInfo queryCudaDeviceInfo() const;

    AppConfig config_;
    Camera camera_;
    CudaDeviceInfo cuda_info_;
    std::unique_ptr<CudaPbo> pbo_;
    std::unique_ptr<GlDisplay> display_;
    GLFWwindow* window_ = nullptr;
    const char* gl_version_text_ = "Unavailable";
    PboSmokeUiState pbo_smoke_ui_;
    bool enable_pbo_smoke_test_ = true;
    int framebuffer_width_ = 0;
    int framebuffer_height_ = 0;
    float animation_time_seconds_ = 0.0f;
    bool imgui_initialized_ = false;
    bool glfw_initialized_ = false;
};

} // namespace vollenia
