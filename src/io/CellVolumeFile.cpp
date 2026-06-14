#include "io/CellVolumeFile.h"

#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <fstream>
#include <stdexcept>
#include <vector>

namespace vollenia {

void CellVolumeFile::loadToDevice(DeviceVolume& output, const std::filesystem::path& path, VolumeDesc desc)
{
    if (!isValidVolumeDesc(desc)) {
        throw std::runtime_error("Cannot load invalid cell volume dimensions");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open cell volume file: " + path.string());
    }

    const std::size_t expected_bytes = volumeByteSize(desc);
    std::vector<float> host(volumeVoxelCount(desc));
    input.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(expected_bytes));
    if (input.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("Cell volume file is smaller than expected: " + path.string());
    }

    char extra = 0;
    if (input.read(&extra, 1)) {
        throw std::runtime_error("Cell volume file is larger than expected: " + path.string());
    }

    output.resize(desc);
    VOL_CUDA_CHECK(cudaMemcpy(output.data(), host.data(), expected_bytes, cudaMemcpyHostToDevice));
}

} // namespace vollenia
