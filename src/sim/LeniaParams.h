#pragma once

#include "core/VolumeDesc.h"

#include <array>
#include <cstddef>

namespace vollenia {

enum class VolumeSource {
    Synthetic = 0,
    Lenia,
};

enum class LeniaParamPreset {
    DiguttomeSaliens = 0,
    DiguttomeTardus,
    TriguttomeLabens,
};

enum class LeniaSeedPreset {
    ReferenceRandomBox = 0,
    CenteredRandomBall,
    AsymmetricGaussianCluster,
    ShellInternalBlobs,
    SmallBlob,
};

enum class KernelCoreType {
    PolynomialBump = 1,
    ExponentialBump = 2,
    Step = 3,
    Staircase = 4,
};

enum class GrowthFunctionType {
    Polynomial = 1,
    Gaussian = 2,
    Step = 3,
};

struct LeniaParams {
    float radius = 10.0f;
    float T = 10.0f;
    float mu = 0.12f;
    float sigma = 0.01f;
    std::array<float, 8> shell_weights {1.0f, 0.75f, 0.5833333f, 0.9166667f, 0.0f, 0.0f, 0.0f, 0.0f};
    int shell_count = 4;
    KernelCoreType kernel_core = KernelCoreType::PolynomialBump;
    GrowthFunctionType growth_function = GrowthFunctionType::Polynomial;
};

struct LeniaStatus {
    bool initialized = false;
    bool kernel_ready = false;
    bool playing = true;
    bool last_step_had_invalid_values = false;
    VolumeDesc desc;
    std::size_t byte_size = 0;
    unsigned long long generation = 0;
    const char* kernel_status = "Not built";
    const char* simulation_status = "Not initialized";
    const char* last_error = "";
};

inline const char* volumeSourceName(VolumeSource source)
{
    switch (source) {
    case VolumeSource::Synthetic:
        return "Synthetic";
    case VolumeSource::Lenia:
        return "Lenia";
    default:
        return "Unknown";
    }
}

inline const char* leniaParamPresetName(LeniaParamPreset preset)
{
    switch (preset) {
    case LeniaParamPreset::DiguttomeSaliens:
        return "Diguttome saliens";
    case LeniaParamPreset::DiguttomeTardus:
        return "Diguttome tardus";
    case LeniaParamPreset::TriguttomeLabens:
        return "Triguttome labens";
    default:
        return "Unknown";
    }
}

inline const char* leniaSeedPresetName(LeniaSeedPreset preset)
{
    switch (preset) {
    case LeniaSeedPreset::ReferenceRandomBox:
        return "Reference random box";
    case LeniaSeedPreset::CenteredRandomBall:
        return "Centered random ball";
    case LeniaSeedPreset::AsymmetricGaussianCluster:
        return "Asymmetric Gaussian cluster";
    case LeniaSeedPreset::ShellInternalBlobs:
        return "Shell + internal blobs";
    case LeniaSeedPreset::SmallBlob:
        return "Small blob";
    default:
        return "Unknown";
    }
}

inline const char* kernelCoreTypeName(KernelCoreType type)
{
    switch (type) {
    case KernelCoreType::PolynomialBump:
        return "Polynomial bump";
    case KernelCoreType::ExponentialBump:
        return "Exponential bump";
    case KernelCoreType::Step:
        return "Step (experimental)";
    case KernelCoreType::Staircase:
        return "Staircase (experimental)";
    default:
        return "Unknown";
    }
}

inline const char* growthFunctionTypeName(GrowthFunctionType type)
{
    switch (type) {
    case GrowthFunctionType::Polynomial:
        return "Polynomial";
    case GrowthFunctionType::Gaussian:
        return "Gaussian";
    case GrowthFunctionType::Step:
        return "Step";
    default:
        return "Unknown";
    }
}

inline LeniaParams leniaParamsForPreset(LeniaParamPreset preset)
{
    switch (preset) {
    case LeniaParamPreset::DiguttomeTardus:
        return LeniaParams {10.0f, 10.0f, 0.15f, 0.016f, {0.6666667f, 1.0f, 0.8333333f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 3, KernelCoreType::PolynomialBump, GrowthFunctionType::Polynomial};
    case LeniaParamPreset::TriguttomeLabens:
        return LeniaParams {10.0f, 10.0f, 0.16f, 0.015f, {1.0f, 0.4166667f, 0.0833333f, 0.1666667f, 0.0f, 0.0f, 0.0f, 0.0f}, 4, KernelCoreType::PolynomialBump, GrowthFunctionType::Polynomial};
    case LeniaParamPreset::DiguttomeSaliens:
    default:
        return LeniaParams {10.0f, 10.0f, 0.12f, 0.01f, {1.0f, 0.75f, 0.5833333f, 0.9166667f, 0.0f, 0.0f, 0.0f, 0.0f}, 4, KernelCoreType::PolynomialBump, GrowthFunctionType::Polynomial};
    }
}

inline bool leniaKernelParamsEqual(const LeniaParams& a, const LeniaParams& b)
{
    if (a.radius != b.radius || a.shell_count != b.shell_count || a.kernel_core != b.kernel_core) {
        return false;
    }
    for (int i = 0; i < a.shell_count; ++i) {
        if (a.shell_weights[i] != b.shell_weights[i]) {
            return false;
        }
    }
    return true;
}

} // namespace vollenia
