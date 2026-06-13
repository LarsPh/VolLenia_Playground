#include "app/App.h"

#include "app/UiPanel.h"
#include "core/CudaCheck.h"
#include "core/GlCheck.h"
#include "render/CudaPbo.h"
#include "render/CudaVolumeRenderer.h"
#include "render/GlDisplay.h"
#include "render/SyntheticVolume.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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

VolumePreset presetFromString(const std::string& value, VolumePreset fallback)
{
    if (value == "sphere") {
        return VolumePreset::Sphere;
    }
    if (value == "shell") {
        return VolumePreset::Shell;
    }
    if (value == "gaussian_blobs") {
        return VolumePreset::GaussianBlobs;
    }
    if (value == "lenia_phantom") {
        return VolumePreset::LeniaPhantom;
    }
    if (value == "axis_ramp") {
        return VolumePreset::AxisRamp;
    }
    return fallback;
}

VolumeRenderMode renderModeFromString(const std::string& value, VolumeRenderMode fallback)
{
    if (value == "emission_absorption") {
        return VolumeRenderMode::EmissionAbsorption;
    }
    if (value == "mip") {
        return VolumeRenderMode::MIP;
    }
    if (value == "first_hit") {
        return VolumeRenderMode::FirstHit;
    }
    return fallback;
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
    render_params_ = config_.render;
    volume_preset_ = config_.preset;
    volume_resolution_ = config_.volume.nx;
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
    synthetic_volume_ = std::make_unique<SyntheticVolume>();
    volume_renderer_ = std::make_unique<CudaVolumeRenderer>();
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

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        handleCameraInput();

        const ImGuiIO& io = ImGui::GetIO();
        UiPanelResult ui_result = ui_panel.render(
            cuda_info_,
            gl_version_text_,
            camera_,
            volume_status_,
            render_enabled_,
            volume_preset_,
            volume_resolution_,
            render_params_,
            io.Framerate,
            frame_time_ms);
        if (ui_result.regenerate_volume) {
            ++volume_seed_;
            volume_dirty_ = true;
            renderer_volume_dirty_ = true;
        }

        VOL_GL_CHECK(glClearColor(0.025f, 0.035f, 0.045f, 1.0f));
        VOL_GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

        try {
            updateFramebufferResources();
            renderVolumeFrame();
        } catch (const std::exception& exception) {
            volume_status_.last_error = exception.what();
            volume_status_.status = "Volume renderer error";
            render_enabled_ = false;
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);

        if (ui_result.quit_requested) {
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
    volume_renderer_.reset();
    synthetic_volume_.reset();
    pbo_.reset();
    display_.reset();
    volume_status_.volume_valid = false;
    volume_status_.texture_valid = false;
    volume_status_.volume_bytes = 0;
}

void App::updateFramebufferResources()
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);

    volume_status_.framebuffer_width = width;
    volume_status_.framebuffer_height = height;
    volume_status_.animation_time_seconds = animation_time_seconds_;

    if (width <= 0 || height <= 0) {
        framebuffer_width_ = width;
        framebuffer_height_ = height;
        volume_status_.status = "Framebuffer minimized; CUDA draw skipped";
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
    }
}

void App::handleCameraInput()
{
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    if (io.MouseDown[0]) {
        camera_.orbit(io.MouseDelta.x, io.MouseDelta.y);
    }
    if (io.MouseWheel != 0.0f) {
        camera_.zoom(io.MouseWheel);
    }
}

void App::renderVolumeFrame()
{
    volume_status_.animation_time_seconds = animation_time_seconds_;
    volume_status_.render_enabled = render_enabled_;
    volume_status_.preset = volume_preset_;

    if (!render_enabled_) {
        volume_status_.status = "Volume renderer disabled";
        return;
    }

    if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) {
        volume_status_.status = "Framebuffer minimized; CUDA draw skipped";
        return;
    }

    if (pbo_ == nullptr || display_ == nullptr || !pbo_->isValid() || !display_->isValid()) {
        volume_status_.status = "PBO/display resources are not ready";
        return;
    }
    if (synthetic_volume_ == nullptr) {
        synthetic_volume_ = std::make_unique<SyntheticVolume>();
    }
    if (volume_renderer_ == nullptr) {
        volume_renderer_ = std::make_unique<CudaVolumeRenderer>();
    }

    if (volume_dirty_) {
        const int resolution = std::clamp(volume_resolution_, 16, 256);
        volume_resolution_ = resolution;
        const VolumeDesc desc {resolution, resolution, resolution};
        synthetic_volume_->resize(desc);
        synthetic_volume_->generate(volume_preset_, volume_seed_);
        volume_dirty_ = false;
        renderer_volume_dirty_ = true;
        volume_status_.status = "Synthetic volume generated";
    }

    if (renderer_volume_dirty_) {
        volume_renderer_->setVolume(*synthetic_volume_);
        renderer_volume_dirty_ = false;
        volume_status_.status = "CUDA 3D texture ready";
    }

    volume_status_.volume_valid = synthetic_volume_->isValid();
    volume_status_.texture_valid = volume_renderer_->hasTexture();
    volume_status_.volume_desc = synthetic_volume_->desc();
    volume_status_.volume_bytes = synthetic_volume_->byteSize();

    bool mapped = false;
    try {
        const PboMapping mapping = pbo_->map();
        mapped = true;

        const float aspect = static_cast<float>(framebuffer_width_) / static_cast<float>(std::max(framebuffer_height_, 1));
        volume_renderer_->render(
            mapping,
            framebuffer_width_,
            framebuffer_height_,
            camera_.frame(aspect),
            render_params_);

        pbo_->unmap();
        mapped = false;

        display_->drawPbo(pbo_->glBuffer(), framebuffer_width_, framebuffer_height_);
        volume_status_.status = "Rendering synthetic volume";
        volume_status_.last_error.clear();
    } catch (...) {
        if (mapped) {
            try {
                pbo_->unmap();
            } catch (const std::exception& exception) {
                volume_status_.last_error = exception.what();
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
    config.camera.yaw_radians = jsonValueOr(camera, "yaw_radians", config.camera.yaw_radians);
    config.camera.pitch_radians = jsonValueOr(camera, "pitch_radians", config.camera.pitch_radians);

    const nlohmann::json render = root.value("render", nlohmann::json::object());
    config.render.step_size = jsonValueOr(render, "step_size", config.render.step_size);
    config.render.density_scale = jsonValueOr(render, "density_scale", config.render.density_scale);
    config.render.threshold = jsonValueOr(render, "threshold", config.render.threshold);
    config.render.brightness = jsonValueOr(render, "brightness", config.render.brightness);
    config.render.max_steps = jsonValueOr(render, "max_steps", config.render.max_steps);
    config.render.early_exit_transmittance = jsonValueOr(render, "early_exit_transmittance", config.render.early_exit_transmittance);
    config.render.mode = renderModeFromString(jsonValueOr(render, "mode", std::string("emission_absorption")), config.render.mode);

    const nlohmann::json volume = root.value("volume", nlohmann::json::object());
    const int resolution = jsonValueOr(volume, "resolution", config.volume.nx);
    config.volume = VolumeDesc {resolution, resolution, resolution};
    config.preset = presetFromString(jsonValueOr(volume, "preset", std::string("lenia_phantom")), config.preset);

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
