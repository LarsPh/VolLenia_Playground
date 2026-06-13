#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace vollenia {

enum class VolumePreset {
    Sphere = 0,
    Shell,
    GaussianBlobs,
    LeniaPhantom,
    AxisRamp,
};

enum class VolumeRenderMode {
    EmissionAbsorption = 0,
    MIP,
    FirstHit,
};

struct VolumeDesc {
    int nx = 128;
    int ny = 128;
    int nz = 128;
};

struct RenderParams {
    float step_size = 0.01f;
    float density_scale = 4.0f;
    float threshold = 0.02f;
    float brightness = 1.0f;
    float early_exit_transmittance = 0.01f;
    int max_steps = 512;
    VolumeRenderMode mode = VolumeRenderMode::EmissionAbsorption;
};

struct CameraFrame {
    float3 position {};
    float3 forward {};
    float3 right {};
    float3 up {};
    float fov_y_degrees = 45.0f;
    float aspect = 1.0f;
};

struct VolumeRenderStatus {
    bool volume_valid = false;
    bool texture_valid = false;
    bool render_enabled = true;
    VolumePreset preset = VolumePreset::Sphere;
    VolumeDesc volume_desc;
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    std::size_t volume_bytes = 0;
    float animation_time_seconds = 0.0f;
    std::string status = "Not initialized";
    std::string last_error;
};

inline const char* volumePresetName(VolumePreset preset)
{
    switch (preset) {
    case VolumePreset::Sphere:
        return "Sphere";
    case VolumePreset::Shell:
        return "Shell";
    case VolumePreset::GaussianBlobs:
        return "Gaussian blobs";
    case VolumePreset::LeniaPhantom:
        return "Lenia-like phantom";
    case VolumePreset::AxisRamp:
        return "Axis ramp";
    default:
        return "Unknown";
    }
}

inline const char* volumeRenderModeName(VolumeRenderMode mode)
{
    switch (mode) {
    case VolumeRenderMode::EmissionAbsorption:
        return "Emission absorption";
    case VolumeRenderMode::MIP:
        return "MIP";
    case VolumeRenderMode::FirstHit:
        return "First hit";
    default:
        return "Unknown";
    }
}

} // namespace vollenia
