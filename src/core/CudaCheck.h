#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace vollenia {

inline void checkCuda(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA error at " << file << ":" << line << " while running " << expression
            << ": " << cudaGetErrorString(result) << " (" << static_cast<int>(result) << ")";
    throw std::runtime_error(message.str());
}

inline std::string cudaVersionToString(int version)
{
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

} // namespace vollenia

#define VOL_CUDA_CHECK(expression) \
    ::vollenia::checkCuda((expression), #expression, __FILE__, __LINE__)
