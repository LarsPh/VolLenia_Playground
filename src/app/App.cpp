#include "app/App.h"

#include "app/UiPanel.h"
#include "core/CudaCheck.h"
#include "core/GlCheck.h"
#include "io/CellResampler.h"
#include "io/CellVolumeFile.h"
#include "render/CudaPbo.h"
#include "render/CudaVolumeRenderer.h"
#include "render/GlDisplay.h"
#include "render/SyntheticVolume.h"
#include "sim/DeviceVolume.h"
#include "sim/LeniaSimulation.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <nlohmann/json.hpp>
#include <nfd.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

VolumeSource sourceFromString(const std::string& value, VolumeSource fallback)
{
    if (value == "synthetic") {
        return VolumeSource::Synthetic;
    }
    if (value == "lenia") {
        return VolumeSource::Lenia;
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

LeniaSeedPreset leniaSeedPresetFromString(const std::string& value, LeniaSeedPreset fallback)
{
    if (value == "reference_random_box") {
        return LeniaSeedPreset::ReferenceRandomBox;
    }
    if (value == "centered_random_ball") {
        return LeniaSeedPreset::CenteredRandomBall;
    }
    if (value == "asymmetric_gaussian_cluster") {
        return LeniaSeedPreset::AsymmetricGaussianCluster;
    }
    if (value == "shell_internal_blobs") {
        return LeniaSeedPreset::ShellInternalBlobs;
    }
    if (value == "small_blob") {
        return LeniaSeedPreset::SmallBlob;
    }
    return fallback;
}

LeniaParamPreset leniaParamPresetFromString(const std::string& value, LeniaParamPreset fallback)
{
    if (value == "diguttome_saliens") {
        return LeniaParamPreset::DiguttomeSaliens;
    }
    if (value == "diguttome_tardus") {
        return LeniaParamPreset::DiguttomeTardus;
    }
    if (value == "triguttome_labens") {
        return LeniaParamPreset::TriguttomeLabens;
    }
    return fallback;
}

KernelCoreType kernelCoreFromInt(int value, KernelCoreType fallback)
{
    switch (value) {
    case 1:
        return KernelCoreType::PolynomialBump;
    case 2:
        return KernelCoreType::ExponentialBump;
    case 3:
        return KernelCoreType::Step;
    case 4:
        return KernelCoreType::Staircase;
    default:
        return fallback;
    }
}

GrowthFunctionType growthFunctionFromInt(int value, GrowthFunctionType fallback)
{
    switch (value) {
    case 1:
        return GrowthFunctionType::Polynomial;
    case 2:
        return GrowthFunctionType::Gaussian;
    case 3:
        return GrowthFunctionType::Step;
    default:
        return fallback;
    }
}

CellResampleMode cellResampleModeFromString(const std::string& value, CellResampleMode fallback)
{
    if (value == "nearest") {
        return CellResampleMode::Nearest;
    }
    if (value == "trilinear") {
        return CellResampleMode::Trilinear;
    }
    return fallback;
}

void configureImGuiFonts(ImGuiIO& io)
{
    static ImVector<ImWchar> cjk_ranges;
    if (cjk_ranges.empty()) {
        ImFontGlyphRangesBuilder builder;
        builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
        builder.AddRanges(io.Fonts->GetGlyphRangesChineseFull());
        builder.AddRanges(io.Fonts->GetGlyphRangesJapanese());
        builder.BuildRanges(&cjk_ranges);
    }

    constexpr std::array<const char*, 3> font_candidates {
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    };
    for (const char* font_path : font_candidates) {
        if (!std::filesystem::exists(font_path)) {
            continue;
        }
        if (io.Fonts->AddFontFromFileTTF(font_path, 16.0f, nullptr, cjk_ranges.Data) != nullptr) {
            return;
        }
    }

    io.Fonts->AddFontDefault();
}

std::string pathToUtf8String(const std::filesystem::path& path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

std::filesystem::path utf8StringToPath(const char* text)
{
    if (text == nullptr) {
        return {};
    }
    const std::string bytes(text);
    return std::filesystem::path(std::u8string(
        reinterpret_cast<const char8_t*>(bytes.data()),
        reinterpret_cast<const char8_t*>(bytes.data() + bytes.size())));
}

std::filesystem::path catalogDialogDefaultPath(const std::filesystem::path& current_path)
{
    const std::filesystem::path diff_bridge_path("outputs/diff_bridge");
    if (std::filesystem::exists(diff_bridge_path)) {
        return diff_bridge_path;
    }
    const std::filesystem::path parent_path = current_path.parent_path();
    if (!parent_path.empty() && std::filesystem::exists(parent_path)) {
        return parent_path;
    }
    return std::filesystem::current_path();
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
    volume_source_ = config_.source;
    loadLeniaConfig("configs/lenia.default.json");
    camera_.setSettings(config_.camera);
    render_params_ = config_.render;
    volume_preset_ = config_.preset;
    volume_resolution_ = config_.volume.nx;
    cuda_info_ = queryCudaDeviceInfo();
    animal_catalog_.load(lenia_config_.animal_catalog_path);

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

    if (NFD_Init() == NFD_OKAY) {
        nfd_initialized_ = true;
    } else {
        const char* error = NFD_GetError();
        animal_catalog_error_ = std::string("Native file picker init failed: ") + (error != nullptr ? error : "unknown error");
    }

    pbo_ = std::make_unique<CudaPbo>();
    display_ = std::make_unique<GlDisplay>();
    synthetic_volume_ = std::make_unique<SyntheticVolume>();
    lenia_simulation_ = std::make_unique<LeniaSimulation>();
    volume_renderer_ = std::make_unique<CudaVolumeRenderer>();
    framebuffer_width_ = framebuffer_width;
    framebuffer_height_ = framebuffer_height;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    configureImGuiFonts(io);
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
            lenia_status_,
            volume_source_,
            animal_catalog_,
            animal_catalog_error_,
            render_enabled_,
            volume_preset_,
            volume_resolution_,
            lenia_config_,
            render_params_,
            io.Framerate,
            frame_time_ms);
        if (ui_result.source_changed) {
            renderer_volume_dirty_ = true;
        }
        if (ui_result.regenerate_volume) {
            ++volume_seed_;
            volume_dirty_ = true;
            renderer_volume_dirty_ = true;
        }
        if (ui_result.lenia_resolution_changed) {
            lenia_sim_dirty_ = true;
            renderer_volume_dirty_ = true;
            if (lenia_config_.cells_source == LeniaCellsSource::Imported) {
                lenia_imported_cells_dirty_ = imported_cells_ != nullptr;
            } else {
                lenia_seed_dirty_ = true;
            }
        }
        if (ui_result.lenia_seed_preset_changed || ui_result.lenia_reset_seed) {
            lenia_config_.cells_source = LeniaCellsSource::Procedural;
            lenia_config_.cells_source_animal_index = -1;
            lenia_config_.imported_cells_scaled = false;
            lenia_seed_dirty_ = true;
        }
        if (ui_result.lenia_regenerate_seed) {
            ++lenia_config_.seed;
            lenia_config_.cells_source = LeniaCellsSource::Procedural;
            lenia_config_.cells_source_animal_index = -1;
            lenia_config_.imported_cells_scaled = false;
            lenia_seed_dirty_ = true;
        }
        if (ui_result.lenia_param_preset_changed || ui_result.lenia_rebuild_kernel) {
            lenia_kernel_dirty_ = true;
        }
        if (ui_result.lenia_single_step) {
            lenia_single_step_requested_ = true;
        }
        if (ui_result.lenia_reload_catalog) {
            loadAnimalCatalogRuntime(lenia_config_.animal_catalog_path);
        }
        if (ui_result.lenia_open_catalog_dialog) {
            openAnimalCatalogDialog();
        }
        if (ui_result.lenia_load_animal) {
            loadAnimalNative(lenia_config_.selected_animal_index);
            volume_source_ = VolumeSource::Lenia;
            renderer_volume_dirty_ = true;
        } else if (ui_result.lenia_load_scaled_animal) {
            loadAnimalCells(lenia_config_.selected_animal_index, true);
            applyAnimalParams(lenia_config_.selected_animal_index, lenia_config_.auto_scale_imported_R);
            volume_source_ = VolumeSource::Lenia;
            renderer_volume_dirty_ = true;
        } else {
            if (ui_result.lenia_apply_cells_only) {
                loadAnimalCells(lenia_config_.selected_animal_index, false);
                volume_source_ = VolumeSource::Lenia;
                renderer_volume_dirty_ = true;
            }
            if (ui_result.lenia_apply_scaled_cells_only) {
                loadAnimalCells(lenia_config_.selected_animal_index, true);
                volume_source_ = VolumeSource::Lenia;
                renderer_volume_dirty_ = true;
            }
            if (ui_result.lenia_apply_params_only) {
                applyAnimalParams(lenia_config_.selected_animal_index, false);
                volume_source_ = VolumeSource::Lenia;
                renderer_volume_dirty_ = true;
            }
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

    if (nfd_initialized_) {
        NFD_Quit();
        nfd_initialized_ = false;
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
    lenia_simulation_.reset();
    scaled_imported_cells_.reset();
    imported_cells_.reset();
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
    if (volume_renderer_ == nullptr) {
        volume_renderer_ = std::make_unique<CudaVolumeRenderer>();
    }

    if (volume_source_ == VolumeSource::Lenia) {
        renderLeniaVolumeFrame();
    } else {
        renderSyntheticVolumeFrame();
    }
}

void App::renderSyntheticVolumeFrame()
{
    if (synthetic_volume_ == nullptr) {
        synthetic_volume_ = std::make_unique<SyntheticVolume>();
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

    drawUploadedVolume(synthetic_volume_->view(), "Rendering synthetic volume");
    renderer_volume_dirty_ = false;
}

void App::renderLeniaVolumeFrame()
{
    if (lenia_simulation_ == nullptr) {
        lenia_simulation_ = std::make_unique<LeniaSimulation>();
    }

    const int resolution = std::clamp(lenia_config_.resolution, 16, 256);
    lenia_config_.resolution = resolution;
    const VolumeDesc desc {resolution, resolution, resolution};

    if (lenia_sim_dirty_ || !lenia_simulation_->isInitialized()) {
        lenia_simulation_->initialize(desc, lenia_config_.params, lenia_config_.seed_preset, lenia_config_.seed);
        lenia_sim_dirty_ = false;
        lenia_seed_dirty_ = false;
        lenia_kernel_dirty_ = false;
    } else {
        lenia_simulation_->setParams(lenia_config_.params);
        if (lenia_seed_dirty_) {
            lenia_simulation_->resetSeed(lenia_config_.seed_preset, lenia_config_.seed);
            lenia_seed_dirty_ = false;
        }
        if (lenia_kernel_dirty_ || lenia_simulation_->kernelDirty()) {
            lenia_simulation_->rebuildKernel();
            lenia_kernel_dirty_ = false;
        }
    }
    if (lenia_imported_cells_dirty_ && imported_cells_ != nullptr) {
        DeviceVolume* cells = scaled_imported_cells_ != nullptr ? scaled_imported_cells_.get() : imported_cells_.get();
        lenia_simulation_->resetImportedCells(cells->view());
        lenia_imported_cells_dirty_ = false;
        lenia_seed_dirty_ = false;
    }

    int steps = 0;
    if (lenia_config_.playing) {
        steps += std::clamp(lenia_config_.steps_per_frame, 0, 32);
    }
    if (lenia_single_step_requested_) {
        ++steps;
        lenia_single_step_requested_ = false;
    }
    if (steps > 0) {
        lenia_simulation_->simulateSteps(steps);
    }

    lenia_status_ = lenia_simulation_->status();
    lenia_status_.playing = lenia_config_.playing;
    if (lenia_status_.last_step_had_invalid_values) {
        lenia_config_.playing = false;
        lenia_status_.playing = false;
    }

    drawUploadedVolume(lenia_simulation_->currentStateView(), "Rendering Lenia simulation");
    lenia_status_ = lenia_simulation_->status();
    lenia_status_.playing = lenia_config_.playing;
}

void App::openAnimalCatalogDialog()
{
    if (!nfd_initialized_) {
        animal_catalog_error_ = "Native file picker is not initialized.";
        return;
    }

    nfdu8char_t* out_path = nullptr;
    const nfdu8filteritem_t filters[] = {
        {"JSON catalog", "json"},
    };
    const std::filesystem::path default_path = catalogDialogDefaultPath(lenia_config_.animal_catalog_path);
    const std::string default_path_text = pathToUtf8String(default_path);
    nfdopendialogu8args_t args {};
    args.filterList = filters;
    args.filterCount = 1;
    args.defaultPath = default_path_text.c_str();

    const nfdresult_t result = NFD_OpenDialogU8_With(&out_path, &args);
    if (result == NFD_OKAY) {
        loadAnimalCatalogRuntime(utf8StringToPath(out_path));
        NFD_FreePathU8(out_path);
        return;
    }
    if (result == NFD_CANCEL) {
        return;
    }

    const char* error = NFD_GetError();
    animal_catalog_error_ = std::string("Native file picker failed: ") + (error != nullptr ? error : "unknown error");
}

bool App::loadAnimalCatalogRuntime(const std::filesystem::path& manifest_path)
{
    LeniaAnimalCatalog next_catalog;
    next_catalog.load(manifest_path);
    if (!next_catalog.isLoaded() || next_catalog.count() <= 0) {
        animal_catalog_error_ = next_catalog.lastError();
        if (animal_catalog_error_.empty()) {
            animal_catalog_error_ = "Catalog has no compatible animal entries: " + manifest_path.string();
        }
        return false;
    }

    animal_catalog_ = std::move(next_catalog);
    lenia_config_.animal_catalog_path = animal_catalog_.manifestPath().string();
    lenia_config_.selected_animal_index = 0;
    animal_catalog_error_.clear();
    return true;
}

void App::loadAnimalNative(int animal_index)
{
    if (animal_catalog_.count() <= 0) {
        volume_status_.last_error = "No Lenia animal catalog is loaded.";
        return;
    }

    const LeniaAnimalPreset& animal = animal_catalog_.animal(animal_index);
    const bool cubic = animal.simulation_desc.nx == animal.simulation_desc.ny
        && animal.simulation_desc.nx == animal.simulation_desc.nz;
    if (cubic && animal.simulation_desc.nx > 0) {
        const int native_resolution = std::clamp(animal.simulation_desc.nx, 16, 256);
        if (lenia_config_.resolution != native_resolution) {
            lenia_config_.resolution = native_resolution;
            lenia_sim_dirty_ = true;
        }
    }

    loadAnimalCells(animal_index, false);
    applyAnimalParams(animal_index, false);
}

void App::loadAnimalCells(int animal_index, bool scaled)
{
    if (animal_catalog_.count() <= 0) {
        volume_status_.last_error = "No Lenia animal catalog is loaded.";
        return;
    }

    const LeniaAnimalPreset& animal = animal_catalog_.animal(animal_index);
    if (imported_cells_ == nullptr) {
        imported_cells_ = std::make_unique<DeviceVolume>();
    }
    CellVolumeFile::loadToDevice(*imported_cells_, animal.cells_file, animal.cells_desc);
    if (scaled) {
        if (scaled_imported_cells_ == nullptr) {
            scaled_imported_cells_ = std::make_unique<DeviceVolume>();
        }
        CellResampler::resampleToDevice(
            *scaled_imported_cells_,
            imported_cells_->view(),
            std::clamp(lenia_config_.imported_cell_scale, 1.0f, 8.0f),
            lenia_config_.cell_resample_mode);
    } else {
        scaled_imported_cells_.reset();
    }
    lenia_config_.imported_cells_scaled = scaled;
    lenia_config_.selected_animal_index = animal_index;
    lenia_config_.cells_source = LeniaCellsSource::Imported;
    lenia_config_.cells_source_animal_index = animal_index;
    lenia_imported_cells_dirty_ = true;
}

void App::applyAnimalParams(int animal_index, bool scale_radius)
{
    if (animal_catalog_.count() <= 0) {
        volume_status_.last_error = "No Lenia animal catalog is loaded.";
        return;
    }

    const LeniaAnimalPreset& animal = animal_catalog_.animal(animal_index);
    lenia_config_.selected_animal_index = animal_index;
    lenia_config_.params = animal.params;
    if (scale_radius) {
        lenia_config_.params.radius *= std::clamp(lenia_config_.imported_cell_scale, 1.0f, 8.0f);
    }
    lenia_config_.params_source = LeniaParamsSource::Animal;
    lenia_config_.params_source_animal_index = animal_index;
    lenia_kernel_dirty_ = true;
}

void App::drawUploadedVolume(DeviceVolumeView volume, const char* status_text)
{
    volume_renderer_->uploadVolume(volume);
    volume_status_.volume_valid = volume.data != nullptr && isValidVolumeDesc(volume.desc);
    volume_status_.texture_valid = volume_renderer_->hasTexture();
    volume_status_.volume_desc = volume.desc;
    volume_status_.volume_bytes = volumeByteSize(volume.desc);

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
        volume_status_.status = status_text;
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
    config.source = sourceFromString(jsonValueOr(root, "source", std::string("lenia")), config.source);

    return config;
}

void App::loadLeniaConfig(const std::string& path)
{
    const std::filesystem::path config_path(path);
    if (!std::filesystem::exists(config_path)) {
        return;
    }

    std::ifstream input(config_path);
    if (!input) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    nlohmann::json root;
    input >> root;

    volume_source_ = sourceFromString(jsonValueOr(root, "source", std::string(volumeSourceName(volume_source_))), volume_source_);

    const nlohmann::json lenia = root.value("lenia", nlohmann::json::object());
    lenia_config_.playing = jsonValueOr(lenia, "playing", lenia_config_.playing);
    lenia_config_.steps_per_frame = std::clamp(jsonValueOr(lenia, "steps_per_frame", lenia_config_.steps_per_frame), 0, 32);
    lenia_config_.resolution = jsonValueOr(lenia, "resolution", lenia_config_.resolution);
    lenia_config_.seed = jsonValueOr(lenia, "seed", lenia_config_.seed);
    lenia_config_.imported_cell_scale = std::clamp(jsonValueOr(lenia, "imported_cell_scale", lenia_config_.imported_cell_scale), 1.0f, 8.0f);
    lenia_config_.auto_scale_imported_R = jsonValueOr(lenia, "auto_scale_imported_R", lenia_config_.auto_scale_imported_R);
    lenia_config_.animal_catalog_path = jsonValueOr(lenia, "animal_catalog", lenia_config_.animal_catalog_path);
    lenia_config_.cell_resample_mode = cellResampleModeFromString(
        jsonValueOr(lenia, "cell_resample_mode", std::string("trilinear")),
        lenia_config_.cell_resample_mode);
    lenia_config_.seed_preset = leniaSeedPresetFromString(
        jsonValueOr(lenia, "seed_preset", std::string("reference_random_box")),
        lenia_config_.seed_preset);
    lenia_config_.param_preset = leniaParamPresetFromString(
        jsonValueOr(lenia, "parameter_preset", std::string("diguttome_saliens")),
        lenia_config_.param_preset);
    lenia_config_.params = leniaParamsForPreset(lenia_config_.param_preset);

    const nlohmann::json params = lenia.value("params", nlohmann::json::object());
    lenia_config_.params.radius = jsonValueOr(params, "R", lenia_config_.params.radius);
    lenia_config_.params.T = jsonValueOr(params, "T", lenia_config_.params.T);
    lenia_config_.params.mu = jsonValueOr(params, "mu", lenia_config_.params.mu);
    lenia_config_.params.sigma = jsonValueOr(params, "sigma", lenia_config_.params.sigma);
    lenia_config_.params.kernel_core = kernelCoreFromInt(jsonValueOr(params, "kn", static_cast<int>(lenia_config_.params.kernel_core)), lenia_config_.params.kernel_core);
    lenia_config_.params.growth_function = growthFunctionFromInt(jsonValueOr(params, "gn", static_cast<int>(lenia_config_.params.growth_function)), lenia_config_.params.growth_function);
    if (params.is_object() && params.contains("b") && params.at("b").is_array()) {
        int shell_count = 0;
        for (const nlohmann::json& value : params.at("b")) {
            if (shell_count >= static_cast<int>(lenia_config_.params.shell_weights.size())) {
                break;
            }
            lenia_config_.params.shell_weights[static_cast<std::size_t>(shell_count)] = value.get<float>();
            ++shell_count;
        }
        if (shell_count > 0) {
            lenia_config_.params.shell_count = shell_count;
        }
    }
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
