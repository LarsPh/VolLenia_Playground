#pragma once

#include "core/VolumeDesc.h"
#include "sim/DeviceVolume.h"

#include <filesystem>

namespace vollenia {

class CellVolumeFile {
public:
    static void loadToDevice(DeviceVolume& output, const std::filesystem::path& path, VolumeDesc desc);
};

} // namespace vollenia
