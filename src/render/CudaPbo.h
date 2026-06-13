#pragma once

#include <cuda_runtime.h>
#include <glad/gl.h>

#include <cstddef>

struct cudaGraphicsResource;

namespace vollenia {

struct PboMapping {
    uchar4* device_ptr = nullptr;
    std::size_t byte_count = 0;
};

class CudaPbo {
public:
    CudaPbo() = default;
    ~CudaPbo() noexcept;

    CudaPbo(const CudaPbo&) = delete;
    CudaPbo& operator=(const CudaPbo&) = delete;

    void resize(int width, int height);
    [[nodiscard]] PboMapping map();
    void unmap();
    void destroy();

    [[nodiscard]] GLuint glBuffer() const { return pbo_; }
    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }
    [[nodiscard]] std::size_t byteSize() const { return byte_size_; }
    [[nodiscard]] bool isValid() const { return pbo_ != 0 && resource_ != nullptr; }
    [[nodiscard]] bool isMapped() const { return mapped_; }

private:
    void destroyNoThrow() noexcept;

    GLuint pbo_ = 0;
    cudaGraphicsResource* resource_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    std::size_t byte_size_ = 0;
    bool mapped_ = false;
};

} // namespace vollenia
