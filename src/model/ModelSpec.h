#pragma once

#include "core/VolumeDesc.h"
#include "model/KernelSpec.h"

#include <filesystem>
#include <string>
#include <vector>

namespace vollenia {

struct FlowParams {
    float theta_A = 1.0f;
    float alpha_power = 2.0f;
    float flow_max = 1.0f;
    float transport_sigma = 0.5f;
    int reintegration_dd = 1;
    FlowBorder border = FlowBorder::Torus;
    FlowGradient gradient = FlowGradient::Sobel3D;
};

struct ModelSpec {
    int format_version = 1;
    std::string model_type = "expanded_flow";
    std::string name = "model";
    VolumeDesc default_desc {64, 64, 64};
    std::vector<ChannelSpec> channels;
    int render_channel = 0;
    ModelUpdateMode update_mode = ModelUpdateMode::ExpandedAdditive;
    float dt = 0.1f;
    bool hard_clip = true;
    FlowParams flow;
    std::vector<KernelSpec> kernels;
    std::filesystem::path source_path;

    [[nodiscard]] int channelCount() const { return static_cast<int>(channels.size()); }
    [[nodiscard]] int kernelCount() const { return static_cast<int>(kernels.size()); }
    [[nodiscard]] bool isMatterChannel(int index) const;
};

class ModelSpecLoader {
public:
    static ModelSpec load(const std::filesystem::path& path);
};

} // namespace vollenia
