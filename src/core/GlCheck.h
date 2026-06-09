#pragma once

#include <glad/gl.h>

#include <sstream>
#include <stdexcept>

namespace vollenia {

inline const char* glErrorName(GLenum error)
{
    switch (error) {
    case GL_NO_ERROR:
        return "GL_NO_ERROR";
    case GL_INVALID_ENUM:
        return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
        return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
        return "GL_INVALID_OPERATION";
    case GL_OUT_OF_MEMORY:
        return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "GL_INVALID_FRAMEBUFFER_OPERATION";
    default:
        return "Unknown OpenGL error";
    }
}

inline void checkGlError(const char* expression, const char* file, int line)
{
    const GLenum error = glGetError();
    if (error == GL_NO_ERROR) {
        return;
    }

    std::ostringstream message;
    message << "OpenGL error at " << file << ":" << line << " after " << expression << ": "
            << glErrorName(error) << " (0x" << std::hex << error << ")";
    throw std::runtime_error(message.str());
}

} // namespace vollenia

#define VOL_GL_CHECK(expression) \
    do {                         \
        (expression);            \
        ::vollenia::checkGlError(#expression, __FILE__, __LINE__); \
    } while (false)
