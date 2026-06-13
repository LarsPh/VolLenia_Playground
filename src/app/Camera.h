#pragma once

#include "render/RenderParams.h"

namespace vollenia {

struct CameraSettings {
    float distance = 3.0f;
    float fov_y_degrees = 45.0f;
    float yaw_radians = 0.0f;
    float pitch_radians = 0.25f;
};

class Camera {
public:
    Camera() = default;
    explicit Camera(CameraSettings settings);

    [[nodiscard]] const CameraSettings& settings() const;
    void setSettings(CameraSettings settings);
    void reset();
    void orbit(float delta_x_pixels, float delta_y_pixels);
    void zoom(float wheel_delta);
    void setDistance(float distance);
    void setFovYDegrees(float fov_y_degrees);
    [[nodiscard]] CameraFrame frame(float aspect) const;

private:
    CameraSettings defaults_;
    CameraSettings settings_;
};

} // namespace vollenia
