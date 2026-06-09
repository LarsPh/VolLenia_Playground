#include "app/App.h"

#include "app/UiPanel.h"
#include "core/CudaCheck.h"
#include "core/GlCheck.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace vollenia {

namespace {

void glfwErrorCallback(int error, const char* description)
{
    std::cerr << "GLFW error " << error << ": " << description << '\n';
}

void framebufferSizeCallback(GLFWwindow*, int width, int height)
{
    VOL_GL_CHECK(glViewport(0, 0, width, height));
}

template <typename T>
T jsonValueOr(const nlohmann::json& object, const char* key, T fallback)
{
    if (!object.is_object() || !object.contains(key)) {
        return fallback;
    }
    return object.at(key).get<T>();
}

} // namespace

App::~App()
{
    shutdown();
}

int App::run()
{
    initialize();
    mainLoop();
    shutdown();
    return 0;
}

void App::initialize()
{
    config_ = loadConfig("configs/app.default.json");
    camera_.setSettings(config_.camera);
    cuda_info_ = queryCudaDeviceInfo();

    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfw_initialized_ = true;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    window_ = glfwCreateWindow(
        config_.window.width,
        config_.window.height,
        config_.window.title.c_str(),
        nullptr,
        nullptr);
    if (window_ == nullptr) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);

    const int glad_version = gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress));
    if (glad_version == 0) {
        throw std::runtime_error("Failed to load OpenGL functions with glad");
    }

    gl_version_text_ = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
    VOL_GL_CHECK(glViewport(0, 0, framebuffer_width, framebuffer_height));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    if (!ImGui_ImplGlfw_InitForOpenGL(window_, true)) {
        throw std::runtime_error("Failed to initialize ImGui GLFW backend");
    }
    if (!ImGui_ImplOpenGL3_Init("#version 330 core")) {
        throw std::runtime_error("Failed to initialize ImGui OpenGL backend");
    }
    imgui_initialized_ = true;
}

void App::mainLoop()
{
    UiPanel ui_panel;
    double previous_time = glfwGetTime();
    float frame_time_ms = 0.0f;

    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
        glfwPollEvents();

        const double current_time = glfwGetTime();
        frame_time_ms = static_cast<float>((current_time - previous_time) * 1000.0);
        previous_time = current_time;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiIO& io = ImGui::GetIO();
        const bool quit_requested = ui_panel.render(
            cuda_info_,
            gl_version_text_,
            camera_.settings(),
            io.Framerate,
            frame_time_ms);

        VOL_GL_CHECK(glClearColor(0.025f, 0.035f, 0.045f, 1.0f));
        VOL_GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);

        if (quit_requested) {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
    }
}

void App::shutdown() noexcept
{
    if (imgui_initialized_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imgui_initialized_ = false;
    }

    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    if (glfw_initialized_) {
        glfwTerminate();
        glfw_initialized_ = false;
    }
}

AppConfig App::loadConfig(const std::string& path) const
{
    AppConfig config;
    const std::filesystem::path config_path(path);
    if (!std::filesystem::exists(config_path)) {
        return config;
    }

    std::ifstream input(config_path);
    if (!input) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    nlohmann::json root;
    input >> root;

    const nlohmann::json window = root.value("window", nlohmann::json::object());
    config.window.width = jsonValueOr(window, "width", config.window.width);
    config.window.height = jsonValueOr(window, "height", config.window.height);
    config.window.title = jsonValueOr(window, "title", config.window.title);

    const nlohmann::json camera = root.value("camera", nlohmann::json::object());
    config.camera.distance = jsonValueOr(camera, "distance", config.camera.distance);
    config.camera.fov_y_degrees = jsonValueOr(camera, "fov_y_degrees", config.camera.fov_y_degrees);

    return config;
}

CudaDeviceInfo App::queryCudaDeviceInfo() const
{
    CudaDeviceInfo info;

    try {
        int device_count = 0;
        VOL_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            info.error = "No CUDA devices were reported by the runtime.";
            return info;
        }

        VOL_CUDA_CHECK(cudaGetDevice(&info.device_index));

        cudaDeviceProp properties {};
        VOL_CUDA_CHECK(cudaGetDeviceProperties(&properties, info.device_index));
        VOL_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
        VOL_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));

        info.available = true;
        info.name = properties.name;
        info.compute_major = properties.major;
        info.compute_minor = properties.minor;
        info.global_memory_bytes = properties.totalGlobalMem;
    } catch (const std::exception& exception) {
        info.available = false;
        info.error = exception.what();
    }

    return info;
}

} // namespace vollenia
