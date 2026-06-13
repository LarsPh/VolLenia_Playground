#pragma once

#include <cuda_runtime.h>

namespace vollenia {

void launchPboSmokeTest(uchar4* output, int width, int height, float time_seconds);

} // namespace vollenia
