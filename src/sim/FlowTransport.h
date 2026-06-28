#pragma once

#include "core/VolumeDesc.h"
#include "model/ModelSpec.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace vollenia {

void launchClearFloat(float* data, std::size_t count, float value);
void launchCopyFloat(float* dst, const float* src, std::size_t count);
void launchAddScaled(float* dst, const float* src, std::size_t count, float scale);
void launchSobelGradient3D(const float* input, float3* output, VolumeDesc desc);
void launchComputeFlowField(
    float3* flow,
    const float3* grad_u,
    const float3* grad_a_sum,
    const float* a_sum,
    VolumeDesc desc,
    float theta_a,
    float alpha_power,
    float flow_max);
void launchFlowTransportSigmaHalf(
    float* next_channel,
    const float* state_channel,
    const float3* flow_channel,
    VolumeDesc desc,
    float dt,
    float flow_max,
    float transport_sigma,
    int reintegration_dd,
    FlowBorder border);
int flowTransportDd(float dt, float flow_max, float transport_sigma, int reintegration_dd);

} // namespace vollenia
