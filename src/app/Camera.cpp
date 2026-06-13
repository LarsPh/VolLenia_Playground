#include "app/Camera.h"

#include <algorithm>
#include <cmath>

namespace vollenia {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kOrbitSpeed = 0.006f;
constexpr float kMinPitch = -1.45f;
constexpr float kMaxPitch = 1.45f;
constexpr float kMinDistance = 0.75f;
constexpr float kMaxDistance = 12.0f;
constexpr float kMinFov = 15.0f;
constexpr float kMaxFov = 90.0f;

float3 add3(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 sub3(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 mul3(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 cross3(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float3 normalize3(float3 v)
{
    const float length = std::sqrt(std::max(dot3(v, v), 1.0e-8f));
    return mul3(v, 1.0f / length);
}

} // namespace

Camera::Camera(CameraSettings settings)
    : defaults_(settings)
    , settings_(settings)
{
}

const CameraSettings& Camera::settings() const
{
    return settings_;
}

void Camera::setSettings(CameraSettings settings)
{
    settings.distance = std::clamp(settings.distance, kMinDistance, kMaxDistance);
    settings.fov_y_degrees = std::clamp(settings.fov_y_degrees, kMinFov, kMaxFov);
    settings.pitch_radians = std::clamp(settings.pitch_radians, kMinPitch, kMaxPitch);
    defaults_ = settings;
    settings_ = settings;
}

void Camera::reset()
{
    settings_ = defaults_;
}

void Camera::orbit(float delta_x_pixels, float delta_y_pixels)
{
    settings_.yaw_radians -= delta_x_pixels * kOrbitSpeed;
    settings_.pitch_radians = std::clamp(settings_.pitch_radians - delta_y_pixels * kOrbitSpeed, kMinPitch, kMaxPitch);

    if (settings_.yaw_radians > kPi) {
        settings_.yaw_radians -= 2.0f * kPi;
    } else if (settings_.yaw_radians < -kPi) {
        settings_.yaw_radians += 2.0f * kPi;
    }
}

void Camera::zoom(float wheel_delta)
{
    const float zoom_factor = std::pow(0.88f, wheel_delta);
    setDistance(settings_.distance * zoom_factor);
}

void Camera::setDistance(float distance)
{
    settings_.distance = std::clamp(distance, kMinDistance, kMaxDistance);
}

void Camera::setFovYDegrees(float fov_y_degrees)
{
    settings_.fov_y_degrees = std::clamp(fov_y_degrees, kMinFov, kMaxFov);
}

CameraFrame Camera::frame(float aspect) const
{
    const float cos_pitch = std::cos(settings_.pitch_radians);
    const float3 position = make_float3(
        settings_.distance * cos_pitch * std::sin(settings_.yaw_radians),
        settings_.distance * std::sin(settings_.pitch_radians),
        settings_.distance * cos_pitch * std::cos(settings_.yaw_radians));

    const float3 target = make_float3(0.0f, 0.0f, 0.0f);
    const float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
    const float3 forward = normalize3(sub3(target, position));
    const float3 right = normalize3(cross3(forward, world_up));
    const float3 up = normalize3(cross3(right, forward));

    CameraFrame frame;
    frame.position = position;
    frame.forward = forward;
    frame.right = right;
    frame.up = up;
    frame.fov_y_degrees = settings_.fov_y_degrees;
    frame.aspect = std::max(aspect, 0.01f);
    return frame;
}

} // namespace vollenia
