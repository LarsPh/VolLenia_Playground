#pragma once

#include <cstddef>

namespace vollenia {

struct VolumeDesc {
    int nx = 128;
    int ny = 128;
    int nz = 128;
};

struct DeviceVolumeView {
    VolumeDesc desc;
    const float* data = nullptr;
};

inline bool volumeDescEquals(VolumeDesc a, VolumeDesc b)
{
    return a.nx == b.nx && a.ny == b.ny && a.nz == b.nz;
}

inline bool isValidVolumeDesc(VolumeDesc desc)
{
    return desc.nx > 0 && desc.ny > 0 && desc.nz > 0;
}

inline std::size_t volumeVoxelCount(VolumeDesc desc)
{
    return static_cast<std::size_t>(desc.nx)
        * static_cast<std::size_t>(desc.ny)
        * static_cast<std::size_t>(desc.nz);
}

inline std::size_t volumeByteSize(VolumeDesc desc)
{
    return volumeVoxelCount(desc) * sizeof(float);
}

} // namespace vollenia
