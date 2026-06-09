#pragma once

namespace vollenia {

struct CameraSettings {
    float distance = 3.0f;
    float fov_y_degrees = 45.0f;
};

class Camera {
public:
    Camera() = default;
    explicit Camera(CameraSettings settings);

    [[nodiscard]] const CameraSettings& settings() const;
    void setSettings(CameraSettings settings);

private:
    CameraSettings settings_;
};

} // namespace vollenia
