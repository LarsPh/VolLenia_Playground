#pragma once

#include "sim/LeniaParams.h"

#include <filesystem>
#include <string>
#include <vector>

namespace vollenia {

struct LeniaAnimalPreset {
    int id = -1;
    int source_index = -1;
    std::string slug;
    std::string code;
    std::string name;
    std::string display_name;
    std::string cname;
    VolumeDesc cells_desc;
    std::filesystem::path cells_file;
    LeniaParams params;
};

class LeniaAnimalCatalog {
public:
    void load(const std::filesystem::path& manifest_path);

    [[nodiscard]] int count() const { return static_cast<int>(animals_.size()); }
    [[nodiscard]] bool isLoaded() const { return loaded_; }
    [[nodiscard]] const std::string& lastError() const { return last_error_; }
    [[nodiscard]] const std::filesystem::path& manifestPath() const { return manifest_path_; }
    [[nodiscard]] const LeniaAnimalPreset& animal(int index) const;

private:
    std::vector<LeniaAnimalPreset> animals_;
    std::filesystem::path manifest_path_;
    std::string last_error_;
    bool loaded_ = false;
};

} // namespace vollenia
