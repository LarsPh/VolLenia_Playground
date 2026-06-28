#include "io/ModelStateFile.h"

#include <nlohmann/json.hpp>

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

std::filesystem::path resolveSiblingPath(const std::filesystem::path& manifest_path, const std::string& value)
{
    std::filesystem::path path(value);
    if (path.is_absolute()) {
        return path;
    }
    return manifest_path.parent_path() / path;
}

} // namespace

ModelStateFile ModelStateLoader::load(const std::filesystem::path& manifest_path)
{
    std::ifstream manifest_input(manifest_path);
    if (!manifest_input) {
        throw std::runtime_error("Failed to open ModelState manifest: " + manifest_path.string());
    }

    nlohmann::json root;
    manifest_input >> root;

    ModelStateFile state;
    state.manifest_path = manifest_path;
    state.format_version = jsonValueOr(root, "format_version", state.format_version);
    if (state.format_version != 1) {
        throw std::runtime_error("Unsupported ModelState format_version: " + std::to_string(state.format_version));
    }
    const std::string layout = jsonValueOr(root, "layout", std::string());
    if (layout != "channel-major-x-fastest") {
        throw std::runtime_error("Unsupported ModelState layout: " + layout);
    }

    if (!root.contains("model_spec") || !root.contains("state_file")) {
        throw std::runtime_error("ModelState requires model_spec and state_file");
    }
    state.model_spec_path = resolveSiblingPath(manifest_path, root.at("model_spec").get<std::string>());
    state.state_path = resolveSiblingPath(manifest_path, root.at("state_file").get<std::string>());

    const nlohmann::json dims = root.value("dims", nlohmann::json::array());
    if (!dims.is_array() || dims.size() != 3) {
        throw std::runtime_error("ModelState dims must be [nx, ny, nz]");
    }
    state.desc = VolumeDesc {dims.at(0).get<int>(), dims.at(1).get<int>(), dims.at(2).get<int>()};
    if (!isValidVolumeDesc(state.desc)) {
        throw std::runtime_error("ModelState dimensions must be positive");
    }
    state.channels = jsonValueOr(root, "channels", state.channels);
    if (state.channels <= 0) {
        throw std::runtime_error("ModelState channels must be positive");
    }

    const nlohmann::json render = root.value("render", nlohmann::json::object());
    state.composite = jsonValueOr(render, "composite", state.composite);
    state.render_channel = jsonValueOr(render, "render_channel", state.render_channel);

    const std::size_t expected_values = volumeVoxelCount(state.desc) * static_cast<std::size_t>(state.channels);
    const std::size_t expected_bytes = expected_values * sizeof(float);
    state.values.resize(expected_values);

    std::ifstream state_input(state.state_path, std::ios::binary);
    if (!state_input) {
        throw std::runtime_error("Failed to open ModelState data: " + state.state_path.string());
    }
    state_input.read(reinterpret_cast<char*>(state.values.data()), static_cast<std::streamsize>(expected_bytes));
    if (state_input.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("ModelState data is smaller than expected: " + state.state_path.string());
    }
    char extra = 0;
    if (state_input.read(&extra, 1)) {
        throw std::runtime_error("ModelState data is larger than expected: " + state.state_path.string());
    }
    return state;
}

} // namespace vollenia
