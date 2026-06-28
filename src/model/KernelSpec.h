#pragma once

#include <string>
#include <vector>

namespace vollenia {

enum class ChannelRole {
    Matter = 0,
    HiddenReserved,
    StaticEnv,
    RenderOnly,
};

enum class ModelUpdateMode {
    ExpandedAdditive = 0,
    Flow,
};

enum class KernelFamily {
    LegacyShell = 0,
    SmoothGaussianMixture,
};

enum class GrowthFamily {
    PolynomialLenia3D = 0,
    Gaussian,
};

enum class FlowBorder {
    Torus = 0,
    Wall,
};

enum class FlowGradient {
    Sobel3D = 0,
};

struct ChannelSpec {
    std::string name = "body";
    ChannelRole role = ChannelRole::Matter;
};

struct GrowthSpec {
    GrowthFamily family = GrowthFamily::Gaussian;
    float mu = 0.12f;
    float sigma = 0.015f;
};

struct SmoothKernelBasis {
    float center = 0.5f;
    float width = 0.1f;
    float amplitude = 1.0f;
};

struct KernelSpec {
    std::string name = "kernel";
    int src = 0;
    int dst = 0;
    float weight = 1.0f;
    KernelFamily family = KernelFamily::SmoothGaussianMixture;
    float radius = 12.0f;
    float envelope_sharpness = 10.0f;
    std::vector<SmoothKernelBasis> basis;
    GrowthSpec growth;

    // Legacy-shell debug path. The smooth family ignores these fields.
    std::vector<float> shell_weights;
};

inline const char* channelRoleName(ChannelRole role)
{
    switch (role) {
    case ChannelRole::Matter:
        return "matter";
    case ChannelRole::HiddenReserved:
        return "hidden_reserved";
    case ChannelRole::StaticEnv:
        return "static_env";
    case ChannelRole::RenderOnly:
        return "render_only";
    default:
        return "unknown";
    }
}

inline const char* modelUpdateModeName(ModelUpdateMode mode)
{
    switch (mode) {
    case ModelUpdateMode::ExpandedAdditive:
        return "expanded_additive";
    case ModelUpdateMode::Flow:
        return "flow";
    default:
        return "unknown";
    }
}

inline const char* kernelFamilyName(KernelFamily family)
{
    switch (family) {
    case KernelFamily::LegacyShell:
        return "legacy_shell";
    case KernelFamily::SmoothGaussianMixture:
        return "smooth_gaussian_mixture";
    default:
        return "unknown";
    }
}

inline const char* growthFamilyName(GrowthFamily family)
{
    switch (family) {
    case GrowthFamily::PolynomialLenia3D:
        return "polynomial_lenia3d";
    case GrowthFamily::Gaussian:
        return "gaussian";
    default:
        return "unknown";
    }
}

} // namespace vollenia
