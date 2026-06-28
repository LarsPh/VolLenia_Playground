#pragma once

#include "app/Camera.h"
#include "io/CellResampler.h"
#include "io/LeniaAnimalCatalog.h"
#include "render/CudaVolumeRenderer.h"
#include "render/RenderParams.h"
#include "sim/ExpandedFlowSimulation.h"
#include "sim/LeniaParams.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <memory>
#include <string>

struct GLFWwindow;

namespace vollenia {

class CudaPbo;
class CudaVolumeRenderer;
class DeviceVolume;
class GlDisplay;
class LeniaSimulation;
class SyntheticVolume;

struct WindowConfig {
    int width = 1600;
    int height = 900;
    std::string title = "VolLenia Playground";
};

struct AppConfig {
    WindowConfig window;
    CameraSettings camera;
    RenderParams render;
    VolumeDesc volume;
    VolumePreset preset = VolumePreset::LeniaPhantom;
    VolumeSource source = VolumeSource::ModelSpec;
};

enum class LeniaCellsSource {
    Procedural = 0,
    Imported,
    Modified,
};

enum class LeniaParamsSource {
    ParameterPreset = 0,
    Animal,
    Modified,
};

struct LeniaConfig {
    bool playing = true;
    bool auto_scale_imported_R = true;
    bool imported_cells_scaled = false;
    int steps_per_frame = 1;
    int resolution = 128;
    int selected_animal_index = 0;
    int cells_source_animal_index = -1;
    int params_source_animal_index = -1;
    unsigned int seed = 1;
    float imported_cell_scale = 4.0f;
    std::string animal_catalog_path = "configs/lenia3d_reference/animals.json";
    LeniaSeedPreset seed_preset = LeniaSeedPreset::ReferenceRandomBox;
    LeniaParamPreset param_preset = LeniaParamPreset::DiguttomeSaliens;
    CellResampleMode cell_resample_mode = CellResampleMode::Trilinear;
    LeniaParams params = leniaParamsForPreset(LeniaParamPreset::DiguttomeSaliens);
    LeniaCellsSource cells_source = LeniaCellsSource::Procedural;
    LeniaParamsSource params_source = LeniaParamsSource::ParameterPreset;
};

struct ModelSpecConfig {
    bool playing = true;
    bool staged_dirty = false;
    bool composite_render = false;
    int steps_per_frame = 1;
    int resolution = 64;
    int render_channel = 0;
    int selected_kernel = 0;
    unsigned int seed = 1;
    std::array<bool, kMaxCompositeChannels> composite_enabled {true, true, true, true};
    std::array<float, kMaxCompositeChannels> composite_intensity {1.0f, 1.0f, 0.85f, 0.85f};
    std::string model_path = "configs/modelspec/expanded_single_kernel.json";
    std::string load_error;
    std::string edit_error;
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
    void renderModelSpecVolumeFrame();
    void drawUploadedVolume(DeviceVolumeView volume, const char* status_text);
    void drawUploadedModelVolume(const char* status_text);
    void openAnimalCatalogDialog();
    void openModelSpecDialog();
    bool loadAnimalCatalogRuntime(const std::filesystem::path& manifest_path);
    bool loadModelSpecRuntime(const std::filesystem::path& spec_path);
    void loadAnimalNative(int animal_index);
    void loadAnimalCells(int animal_index, bool scaled);
    void applyAnimalParams(int animal_index, bool scale_radius);

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
    std::unique_ptr<ExpandedFlowSimulation> modelspec_simulation_;
    std::unique_ptr<DeviceVolume> imported_cells_;
    std::unique_ptr<DeviceVolume> scaled_imported_cells_;
    LeniaAnimalCatalog animal_catalog_;
    GLFWwindow* window_ = nullptr;
    const char* gl_version_text_ = "Unavailable";
    RenderParams render_params_;
    VolumeRenderStatus volume_status_;
    std::string animal_catalog_error_;
    LeniaConfig lenia_config_;
    LeniaStatus lenia_status_;
    ModelSpecConfig modelspec_config_;
    ExpandedFlowStatus modelspec_status_;
    ModelSpec modelspec_;
    ModelSpec modelspec_staged_;
    VolumeSource volume_source_ = VolumeSource::ModelSpec;
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
    bool lenia_imported_cells_dirty_ = false;
    bool modelspec_sim_dirty_ = true;
    bool modelspec_single_step_requested_ = false;
    bool modelspec_reload_requested_ = false;
    int framebuffer_width_ = 0;
    int framebuffer_height_ = 0;
    float animation_time_seconds_ = 0.0f;
    bool imgui_initialized_ = false;
    bool glfw_initialized_ = false;
    bool nfd_initialized_ = false;
};

} // namespace vollenia
