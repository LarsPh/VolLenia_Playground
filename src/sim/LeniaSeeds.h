#pragma once

#include "sim/DeviceVolume.h"
#include "sim/LeniaParams.h"

namespace vollenia {

void launchLeniaSeed(DeviceVolume& volume, LeniaSeedPreset preset, unsigned int seed);

} // namespace vollenia
