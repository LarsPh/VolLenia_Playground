#include "app/Camera.h"

namespace vollenia {

Camera::Camera(CameraSettings settings)
    : settings_(settings)
{
}

const CameraSettings& Camera::settings() const
{
    return settings_;
}

void Camera::setSettings(CameraSettings settings)
{
    settings_ = settings;
}

} // namespace vollenia
