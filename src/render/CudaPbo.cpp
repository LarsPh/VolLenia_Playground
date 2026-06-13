#include "render/CudaPbo.h"

#include "core/CudaCheck.h"
#include "core/GlCheck.h"

#include <cuda_gl_interop.h>

#include <iostream>
#include <stdexcept>

namespace vollenia {

namespace {

std::size_t pboByteSize(int width, int height)
{
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * sizeof(uchar4);
}

} // namespace

CudaPbo::~CudaPbo() noexcept
{
    destroyNoThrow();
}

void CudaPbo::resize(int width, int height)
{
    if (width <= 0 || height <= 0) {
        destroy();
        return;
    }

    if (width == width_ && height == height_ && isValid()) {
        return;
    }

    destroy();

    width_ = width;
    height_ = height;
    byte_size_ = pboByteSize(width, height);

    VOL_GL_CHECK(glGenBuffers(1, &pbo_));
    VOL_GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_));
    VOL_GL_CHECK(glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        static_cast<GLsizeiptr>(byte_size_),
        nullptr,
        GL_STREAM_DRAW));
    VOL_GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

    VOL_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &resource_,
        pbo_,
        cudaGraphicsRegisterFlagsWriteDiscard));
}

PboMapping CudaPbo::map()
{
    if (!isValid()) {
        throw std::runtime_error("Cannot map invalid CUDA/OpenGL PBO");
    }
    if (mapped_) {
        throw std::runtime_error("CUDA/OpenGL PBO is already mapped");
    }

    VOL_CUDA_CHECK(cudaGraphicsMapResources(1, &resource_, 0));
    mapped_ = true;

    PboMapping mapping;
    void* device_ptr = nullptr;
    try {
        VOL_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            &device_ptr,
            &mapping.byte_count,
            resource_));
    } catch (...) {
        unmap();
        throw;
    }
    mapping.device_ptr = static_cast<uchar4*>(device_ptr);
    return mapping;
}

void CudaPbo::unmap()
{
    if (!mapped_) {
        return;
    }

    VOL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource_, 0));
    mapped_ = false;
}

void CudaPbo::destroy()
{
    if (mapped_) {
        unmap();
    }

    if (resource_ != nullptr) {
        VOL_CUDA_CHECK(cudaGraphicsUnregisterResource(resource_));
        resource_ = nullptr;
    }

    if (pbo_ != 0) {
        VOL_GL_CHECK(glDeleteBuffers(1, &pbo_));
        pbo_ = 0;
    }

    width_ = 0;
    height_ = 0;
    byte_size_ = 0;
    mapped_ = false;
}

void CudaPbo::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy CUDA/OpenGL PBO cleanly: " << exception.what() << '\n';
    }
}

} // namespace vollenia
