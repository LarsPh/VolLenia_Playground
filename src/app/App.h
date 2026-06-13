#pragma once

#include "app/Camera.h"

#include <cstddef>
#include <string>

struct GLFWwindow;

namespace vollenia {

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

class App {
public:
    App() = default;
    ~App();

    App(const App&) = delete; // disable copy constructor since it holds GLFW window and OpenGL context (possible double free)
    App& operator=(const App&) = delete; // disable copy assignment

    int run();

private:
    void initialize();
    void mainLoop();
    void shutdown() noexcept; // called by destructor

    [[nodiscard]] AppConfig loadConfig(const std::string& path) const; // nodiscard since return something important
    [[nodiscard]] CudaDeviceInfo queryCudaDeviceInfo() const;

    AppConfig config_;
    Camera camera_;
    CudaDeviceInfo cuda_info_;
    GLFWwindow* window_ = nullptr;
    const char* gl_version_text_ = "Unavailable";
    bool imgui_initialized_ = false;
    bool glfw_initialized_ = false;
};

} // namespace vollenia
