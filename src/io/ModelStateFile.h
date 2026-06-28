#pragma once

#include "core/VolumeDesc.h"

#include <filesystem>
#include <string>
#include <vector>

namespace vollenia {

struct ModelStateFile {
    int format_version = 1;
    std::filesystem::path manifest_path;
    std::filesystem::path model_spec_path;
    std::filesystem::path state_path;
    VolumeDesc desc;
    int channels = 0;
    bool composite = true;
    int render_channel = 0;
    std::vector<float> values;
};

class ModelStateLoader {
public:
    static ModelStateFile load(const std::filesystem::path& manifest_path);
};

} // namespace vollenia
