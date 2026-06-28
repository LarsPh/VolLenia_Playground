#include "model/ModelSpec.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace vollenia {

namespace {

template <typename T>
T jsonValueOr(const nlohmann::json& object, const char* key, T fallback)
{
    if (!object.is_object() || !object.contains(key)) {
        return fallback;
    }
    return object.at(key).get<T>();
}

ChannelRole parseChannelRole(const std::string& value)
{
    if (value == "matter") {
        return ChannelRole::Matter;
    }
    if (value == "hidden_reserved") {
        return ChannelRole::HiddenReserved;
    }
    if (value == "static_env") {
        return ChannelRole::StaticEnv;
    }
    if (value == "render_only") {
        return ChannelRole::RenderOnly;
    }
    throw std::runtime_error("Unknown ModelSpec channel role: " + value);
}

ModelUpdateMode parseUpdateMode(const std::string& value)
{
    if (value == "expanded_additive") {
        return ModelUpdateMode::ExpandedAdditive;
    }
    if (value == "flow") {
        return ModelUpdateMode::Flow;
    }
    throw std::runtime_error("Unknown ModelSpec update mode: " + value);
}

KernelFamily parseKernelFamily(const std::string& value)
{
    if (value == "legacy_shell") {
        return KernelFamily::LegacyShell;
    }
    if (value == "smooth_gaussian_mixture") {
        return KernelFamily::SmoothGaussianMixture;
    }
    throw std::runtime_error("Unknown ModelSpec kernel family: " + value);
}

GrowthFamily parseGrowthFamily(const std::string& value)
{
    if (value == "polynomial_lenia3d") {
        return GrowthFamily::PolynomialLenia3D;
    }
    if (value == "gaussian") {
        return GrowthFamily::Gaussian;
    }
    throw std::runtime_error("Unknown ModelSpec growth family: " + value);
}

FlowBorder parseFlowBorder(const std::string& value)
{
    if (value == "torus") {
        return FlowBorder::Torus;
    }
    if (value == "wall") {
        return FlowBorder::Wall;
    }
    throw std::runtime_error("Unsupported ModelSpec flow border: " + value);
}

FlowGradient parseFlowGradient(const std::string& value)
{
    if (value == "sobel3d") {
        return FlowGradient::Sobel3D;
    }
    throw std::runtime_error("Unsupported ModelSpec flow gradient: " + value);
}

VolumeDesc parseOptionalDesc(const nlohmann::json& root, VolumeDesc fallback)
{
    const nlohmann::json simulation = root.value("simulation", nlohmann::json::object());
    const nlohmann::json state = root.value("state", nlohmann::json::object());
    const nlohmann::json dims = simulation.value("dims", state.value("dims", nlohmann::json::array()));
    if (dims.is_array() && dims.size() == 3) {
        return VolumeDesc {dims.at(0).get<int>(), dims.at(1).get<int>(), dims.at(2).get<int>()};
    }
    const int resolution = jsonValueOr(simulation, "resolution", jsonValueOr(state, "resolution", fallback.nx));
    return VolumeDesc {resolution, resolution, resolution};
}

void validateSpec(const ModelSpec& spec)
{
    if (spec.format_version != 1) {
        throw std::runtime_error("Unsupported ModelSpec format_version: " + std::to_string(spec.format_version));
    }
    if (spec.model_type != "expanded_flow") {
        throw std::runtime_error("Unsupported ModelSpec model_type: " + spec.model_type);
    }
    if (!isValidVolumeDesc(spec.default_desc)) {
        throw std::runtime_error("ModelSpec simulation dimensions must be positive");
    }
    if (spec.channels.empty()) {
        throw std::runtime_error("ModelSpec must contain at least one channel");
    }
    if (spec.render_channel < 0 || spec.render_channel >= spec.channelCount()) {
        throw std::runtime_error("ModelSpec render_channel is out of range");
    }
    if (spec.kernels.empty()) {
        throw std::runtime_error("ModelSpec must contain at least one kernel");
    }
    if (spec.dt <= 0.0f) {
        throw std::runtime_error("ModelSpec update.dt must be positive");
    }
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.src < 0 || kernel.src >= spec.channelCount() || kernel.dst < 0 || kernel.dst >= spec.channelCount()) {
            throw std::runtime_error("ModelSpec kernel channel index is out of range: " + kernel.name);
        }
        if (kernel.radius <= 0.0f) {
            throw std::runtime_error("ModelSpec kernel radius must be positive: " + kernel.name);
        }
        if (kernel.family == KernelFamily::SmoothGaussianMixture && kernel.basis.empty()) {
            throw std::runtime_error("Smooth Gaussian-mixture kernel requires at least one basis entry: " + kernel.name);
        }
        if (kernel.growth.sigma <= 0.0f) {
            throw std::runtime_error("ModelSpec growth sigma must be positive: " + kernel.name);
        }
    }
    if (spec.update_mode == ModelUpdateMode::Flow) {
        if (std::abs(spec.flow.transport_sigma - 0.5f) > 1.0e-6f) {
            throw std::runtime_error("Stage 1 Flow supports only transport_sigma = 0.5");
        }
        if (spec.flow.theta_A <= 0.0f || spec.flow.alpha_power <= 0.0f || spec.flow.flow_max <= 0.0f) {
            throw std::runtime_error("ModelSpec flow theta_A, alpha_power, and flow_max must be positive");
        }
    }
}

} // namespace

bool ModelSpec::isMatterChannel(int index) const
{
    return index >= 0 && index < channelCount() && channels[static_cast<std::size_t>(index)].role == ChannelRole::Matter;
}

ModelSpec ModelSpecLoader::load(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open ModelSpec: " + path.string());
    }

    nlohmann::json root;
    input >> root;

    ModelSpec spec;
    spec.source_path = path;
    spec.name = jsonValueOr(root, "name", path.stem().string());
    spec.format_version = jsonValueOr(root, "format_version", spec.format_version);
    spec.model_type = jsonValueOr(root, "model_type", spec.model_type);
    spec.default_desc = parseOptionalDesc(root, spec.default_desc);

    const nlohmann::json state = root.value("state", nlohmann::json::object());
    spec.render_channel = jsonValueOr(state, "render_channel", spec.render_channel);
    const nlohmann::json channels = state.value("channels", nlohmann::json::array());
    for (const nlohmann::json& item : channels) {
        if (!item.is_object()) {
            throw std::runtime_error("ModelSpec channel entries must be objects");
        }
        ChannelSpec channel;
        channel.name = jsonValueOr(item, "name", std::string("channel_") + std::to_string(spec.channels.size()));
        channel.role = parseChannelRole(jsonValueOr(item, "role", std::string("matter")));
        spec.channels.push_back(std::move(channel));
    }

    const nlohmann::json update = root.value("update", nlohmann::json::object());
    spec.update_mode = parseUpdateMode(jsonValueOr(update, "mode", std::string("expanded_additive")));
    spec.dt = jsonValueOr(update, "dt", spec.dt);
    spec.hard_clip = jsonValueOr(update, "clip", std::string("hard")) == "hard";

    const nlohmann::json flow = update.value("flow", nlohmann::json::object());
    spec.flow.theta_A = jsonValueOr(flow, "theta_A", spec.flow.theta_A);
    spec.flow.alpha_power = jsonValueOr(flow, "alpha_power", spec.flow.alpha_power);
    spec.flow.flow_max = jsonValueOr(flow, "flow_max", spec.flow.flow_max);
    spec.flow.transport_sigma = jsonValueOr(flow, "transport_sigma", spec.flow.transport_sigma);
    spec.flow.reintegration_dd = jsonValueOr(flow, "reintegration_dd", spec.flow.reintegration_dd);
    spec.flow.border = parseFlowBorder(jsonValueOr(flow, "border", std::string("torus")));
    spec.flow.gradient = parseFlowGradient(jsonValueOr(flow, "gradient", std::string("sobel3d")));

    const nlohmann::json kernels = root.value("kernels", nlohmann::json::array());
    for (const nlohmann::json& item : kernels) {
        if (!item.is_object()) {
            throw std::runtime_error("ModelSpec kernel entries must be objects");
        }
        KernelSpec kernel;
        kernel.name = jsonValueOr(item, "name", std::string("kernel_") + std::to_string(spec.kernels.size()));
        kernel.src = jsonValueOr(item, "src", kernel.src);
        kernel.dst = jsonValueOr(item, "dst", kernel.dst);
        kernel.weight = jsonValueOr(item, "weight", kernel.weight);
        kernel.family = parseKernelFamily(jsonValueOr(item, "family", std::string("smooth_gaussian_mixture")));
        kernel.radius = jsonValueOr(item, "R", jsonValueOr(item, "radius", kernel.radius));
        kernel.envelope_sharpness = jsonValueOr(item, "envelope_sharpness", kernel.envelope_sharpness);

        const nlohmann::json basis = item.value("basis", nlohmann::json::array());
        for (const nlohmann::json& entry : basis) {
            SmoothKernelBasis parsed;
            parsed.center = jsonValueOr(entry, "center", parsed.center);
            parsed.width = jsonValueOr(entry, "width", parsed.width);
            parsed.amplitude = jsonValueOr(entry, "amplitude", parsed.amplitude);
            kernel.basis.push_back(parsed);
        }
        if (item.contains("shell_weights") && item.at("shell_weights").is_array()) {
            for (const nlohmann::json& value : item.at("shell_weights")) {
                kernel.shell_weights.push_back(value.get<float>());
            }
        }

        const nlohmann::json growth = item.value("growth", nlohmann::json::object());
        kernel.growth.family = parseGrowthFamily(jsonValueOr(growth, "family", std::string("gaussian")));
        kernel.growth.mu = jsonValueOr(growth, "mu", kernel.growth.mu);
        kernel.growth.sigma = jsonValueOr(growth, "sigma", kernel.growth.sigma);
        spec.kernels.push_back(std::move(kernel));
    }

    validateSpec(spec);
    return spec;
}

} // namespace vollenia
