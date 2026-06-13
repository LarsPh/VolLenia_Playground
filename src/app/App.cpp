#include "app/App.h"

#include "app/UiPanel.h"
#include "core/CudaCheck.h"
#include "core/GlCheck.h"
#include "render/CudaPbo.h"
#include "render/GlDisplay.h"
#include "render/PboSmokeTest.h"

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

App::App() = default;

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

    pbo_ = std::make_unique<CudaPbo>();
    display_ = std::make_unique<GlDisplay>();
    framebuffer_width_ = framebuffer_width;
    framebuffer_height_ = framebuffer_height;

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
        const float delta_time_seconds = static_cast<float>(current_time - previous_time);
        frame_time_ms = delta_time_seconds * 1000.0f;
        animation_time_seconds_ += delta_time_seconds;
        previous_time = current_time;

        VOL_GL_CHECK(glClearColor(0.025f, 0.035f, 0.045f, 1.0f));
        VOL_GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

        try {
            updateFramebufferResources();
            renderPboSmokeFrame();
        } catch (const std::exception& exception) {
            pbo_smoke_ui_.last_error = exception.what();
            pbo_smoke_ui_.status = "Interop error; smoke test disabled";
            enable_pbo_smoke_test_ = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiIO& io = ImGui::GetIO();
        const bool quit_requested = ui_panel.render(
            cuda_info_,
            gl_version_text_,
            camera_.settings(),
            pbo_smoke_ui_,
            enable_pbo_smoke_test_,
            io.Framerate,
            frame_time_ms);

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
    destroyRenderResources();

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

void App::destroyRenderResources() noexcept
{
    pbo_.reset();
    display_.reset();
    pbo_smoke_ui_.resource_valid = false;
    pbo_smoke_ui_.pbo_byte_size = 0;
}

void App::updateFramebufferResources()
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);

    pbo_smoke_ui_.framebuffer_width = width;
    pbo_smoke_ui_.framebuffer_height = height;
    pbo_smoke_ui_.animation_time_seconds = animation_time_seconds_;

    if (width <= 0 || height <= 0) {
        framebuffer_width_ = width;
        framebuffer_height_ = height;
        pbo_smoke_ui_.resource_valid = pbo_ != nullptr && pbo_->isValid();
        pbo_smoke_ui_.pbo_byte_size = pbo_ != nullptr ? pbo_->byteSize() : 0;
        pbo_smoke_ui_.status = "Framebuffer minimized; CUDA draw skipped";
        return;
    }

    if (pbo_ == nullptr) {
        pbo_ = std::make_unique<CudaPbo>();
    }
    if (display_ == nullptr) {
        display_ = std::make_unique<GlDisplay>();
    }

    if (width != framebuffer_width_ || height != framebuffer_height_ || !pbo_->isValid() || !display_->isValid()) {
        pbo_->resize(width, height);
        display_->resize(width, height);
        framebuffer_width_ = width;
        framebuffer_height_ = height;
        pbo_smoke_ui_.status = "CUDA/OpenGL interop resources ready";
    }

    pbo_smoke_ui_.resource_valid = pbo_->isValid() && display_->isValid();
    pbo_smoke_ui_.pbo_byte_size = pbo_->byteSize();
}

void App::renderPboSmokeFrame()
{
    pbo_smoke_ui_.animation_time_seconds = animation_time_seconds_;

    if (!enable_pbo_smoke_test_) {
        pbo_smoke_ui_.status = "PBO smoke test disabled";
        return;
    }

    if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
        pbo_smoke_ui_.status = "Framebuffer minimized; CUDA draw skipped";
        return;
    }

    if (pbo_ == nullptr || display_ == nullptr || !pbo_->isValid() || !display_->isValid()) {
        pbo_smoke_ui_.status = "Interop resources are not ready";
        return;
    }

    bool mapped = false;
    try {
        const PboMapping mapping = pbo_->map();
        mapped = true;

        launchPboSmokeTest(
            mapping.device_ptr,
            framebuffer_width_,
            framebuffer_height_,
            animation_time_seconds_);

        pbo_->unmap();
        mapped = false;

        display_->drawPbo(pbo_->glBuffer(), framebuffer_width_, framebuffer_height_);
        pbo_smoke_ui_.status = "Drawing CUDA-generated PBO smoke image";
        pbo_smoke_ui_.last_error.clear();
    } catch (...) {
        if (mapped) {
            try {
                pbo_->unmap();
            } catch (const std::exception& exception) {
                pbo_smoke_ui_.last_error = exception.what();
            }
        }
        throw;
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
