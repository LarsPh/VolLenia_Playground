#include "io/LeniaAnimalCatalog.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <map>
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

KernelCoreType kernelCoreFromId(int id)
{
    switch (id) {
    case 2:
        return KernelCoreType::ExponentialBump;
    case 3:
        return KernelCoreType::Step;
    case 4:
        return KernelCoreType::Staircase;
    case 1:
    default:
        return KernelCoreType::PolynomialBump;
    }
}

GrowthFunctionType growthFunctionFromId(int id)
{
    switch (id) {
    case 2:
        return GrowthFunctionType::Gaussian;
    case 3:
        return GrowthFunctionType::Step;
    case 1:
    default:
        return GrowthFunctionType::Polynomial;
    }
}

std::string fallbackAnimalName(int id)
{
    return "Animal #" + std::to_string(id);
}

VolumeDesc parseVolumeDesc(const nlohmann::json& dims)
{
    if (!dims.is_array() || dims.size() != 3) {
        throw std::runtime_error("Expected dims array [nx, ny, nz]");
    }
    return VolumeDesc {dims.at(0).get<int>(), dims.at(1).get<int>(), dims.at(2).get<int>()};
}

void assignDisplayNames(std::vector<LeniaAnimalPreset>& animals)
{
    std::map<std::string, int> name_counts;
    for (const LeniaAnimalPreset& animal : animals) {
        const std::string base_name = animal.name.empty() ? fallbackAnimalName(animal.id) : animal.name;
        ++name_counts[base_name];
    }

    for (LeniaAnimalPreset& animal : animals) {
        const std::string base_name = animal.name.empty() ? fallbackAnimalName(animal.id) : animal.name;
        animal.display_name = base_name;
        if (name_counts[base_name] > 1) {
            if (!animal.code.empty()) {
                animal.display_name += " [" + animal.code + "]";
            } else {
                animal.display_name += " #" + std::to_string(animal.id);
            }
        }
    }

    std::map<std::string, int> display_name_counts;
    for (const LeniaAnimalPreset& animal : animals) {
        ++display_name_counts[animal.display_name];
    }

    for (LeniaAnimalPreset& animal : animals) {
        if (display_name_counts[animal.display_name] <= 1) {
            continue;
        }
        if (animal.source_index >= 0) {
            animal.display_name += " src #" + std::to_string(animal.source_index);
        } else {
            animal.display_name += " id #" + std::to_string(animal.id);
        }
    }
}

LeniaParams parseParams(const nlohmann::json& params)
{
    LeniaParams parsed;
    parsed.radius = jsonValueOr(params, "R", parsed.radius);
    parsed.T = jsonValueOr(params, "T", parsed.T);
    parsed.mu = jsonValueOr(params, "m", jsonValueOr(params, "mu", parsed.mu));
    parsed.sigma = jsonValueOr(params, "s", jsonValueOr(params, "sigma", parsed.sigma));
    parsed.kernel_core = kernelCoreFromId(jsonValueOr(params, "kn", static_cast<int>(parsed.kernel_core)));
    parsed.growth_function = growthFunctionFromId(jsonValueOr(params, "gn", static_cast<int>(parsed.growth_function)));

    if (params.is_object() && params.contains("b") && params.at("b").is_array()) {
        parsed.shell_weights.fill(0.0f);
        int count = 0;
        for (const nlohmann::json& value : params.at("b")) {
            if (count >= static_cast<int>(parsed.shell_weights.size())) {
                break;
            }
            parsed.shell_weights[static_cast<std::size_t>(count)] = value.get<float>();
            ++count;
        }
        parsed.shell_count = std::max(count, 1);
    }

    return parsed;
}

} // namespace

void LeniaAnimalCatalog::load(const std::filesystem::path& manifest_path)
{
    animals_.clear();
    manifest_path_ = manifest_path;
    last_error_.clear();
    loaded_ = false;

    if (!std::filesystem::exists(manifest_path)) {
        last_error_ = "Catalog manifest not found: " + manifest_path.string();
        return;
    }

    try {
        std::ifstream input(manifest_path);
        if (!input) {
            throw std::runtime_error("Failed to open catalog manifest");
        }

        nlohmann::json root;
        input >> root;
        const nlohmann::json animals = root.value("animals", nlohmann::json::array());
        const std::filesystem::path base_dir = manifest_path.parent_path();

        for (const nlohmann::json& item : animals) {
            if (!item.is_object()) {
                continue;
            }
            const nlohmann::json dims = item.value("dims", nlohmann::json::array());
            if (!dims.is_array() || dims.size() != 3) {
                continue;
            }
            const nlohmann::json simulation_dims = item.value("simulation_dims", dims);

            LeniaAnimalPreset animal;
            animal.id = jsonValueOr(item, "id", static_cast<int>(animals_.size()));
            animal.source_index = jsonValueOr(item, "source_index", -1);
            animal.slug = jsonValueOr(item, "slug", std::string());
            animal.code = jsonValueOr(item, "code", std::string());
            animal.name = jsonValueOr(item, "name", std::string());
            animal.display_name = animal.name.empty() ? fallbackAnimalName(animal.id) : animal.name;
            animal.cname = jsonValueOr(item, "cname", std::string());
            animal.cells_desc = parseVolumeDesc(dims);
            animal.simulation_desc = parseVolumeDesc(simulation_dims);
            animal.resolution_policy = jsonValueOr(item, "resolution_policy", std::string("native"));
            animal.cells_file = (base_dir / jsonValueOr(item, "cells_file", std::string())).lexically_normal();
            animal.params = parseParams(item.value("params", nlohmann::json::object()));
            animals_.push_back(std::move(animal));
        }

        assignDisplayNames(animals_);
        loaded_ = true;
    } catch (const std::exception& exception) {
        animals_.clear();
        last_error_ = exception.what();
        loaded_ = false;
    }
}

const LeniaAnimalPreset& LeniaAnimalCatalog::animal(int index) const
{
    if (index < 0 || index >= count()) {
        throw std::out_of_range("Lenia animal index out of range");
    }
    return animals_[static_cast<std::size_t>(index)];
}

} // namespace vollenia
