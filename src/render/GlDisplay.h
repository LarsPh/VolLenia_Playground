#pragma once

#include <glad/gl.h>

namespace vollenia {

class GlDisplay {
public:
    GlDisplay() = default;
    ~GlDisplay() noexcept;

    GlDisplay(const GlDisplay&) = delete;
    GlDisplay& operator=(const GlDisplay&) = delete;

    void resize(int width, int height);
    void drawPbo(GLuint pbo, int width, int height);
    void destroy();

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }
    [[nodiscard]] bool isValid() const { return texture_ != 0 && vao_ != 0 && program_ != 0; }

private:
    void ensurePipeline();
    void destroyNoThrow() noexcept;

    [[nodiscard]] GLuint compileShader(GLenum type, const char* source) const;
    [[nodiscard]] GLuint createProgram() const;

    GLuint texture_ = 0;
    GLuint vao_ = 0;
    GLuint program_ = 0;
    int width_ = 0;
    int height_ = 0;
};

} // namespace vollenia
