#pragma once

#include "sim/DeviceVolume.h"

namespace vollenia {

enum class CellResampleMode {
    Nearest = 0,
    Trilinear,
};

const char* cellResampleModeName(CellResampleMode mode);

class CellResampler {
public:
    static void resampleToDevice(
        DeviceVolume& output,
        DeviceVolumeView source,
        float scale,
        CellResampleMode mode);
};

} // namespace vollenia
