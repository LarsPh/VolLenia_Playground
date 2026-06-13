#include "render/GlDisplay.h"

#include "core/GlCheck.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace vollenia {

namespace {

constexpr const char* kVertexShader = R"glsl(
#version 330 core
out vec2 v_uv;

void main()
{
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 position = positions[gl_VertexID];
    v_uv = 0.5 * (position + vec2(1.0));
    gl_Position = vec4(position, 0.0, 1.0);
}
)glsl";

constexpr const char* kFragmentShader = R"glsl(
#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_image;

void main()
{
    frag_color = texture(u_image, v_uv);
}
)glsl";

std::string shaderInfoLog(GLuint shader)
{
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return {};
    }

    std::string log(static_cast<std::size_t>(length), '\0');
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    return log;
}

std::string programInfoLog(GLuint program)
{
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return {};
    }

    std::string log(static_cast<std::size_t>(length), '\0');
    glGetProgramInfoLog(program, length, nullptr, log.data());
    return log;
}

} // namespace

GlDisplay::~GlDisplay() noexcept
{
    destroyNoThrow();
}

void GlDisplay::resize(int width, int height)
{
    if (width <= 0 || height <= 0) {
        destroy();
        return;
    }

    ensurePipeline();

    if (texture_ != 0 && width == width_ && height == height_) {
        return;
    }

    if (texture_ == 0) {
        VOL_GL_CHECK(glGenTextures(1, &texture_));
    }

    width_ = width;
    height_ = height;

    VOL_GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_));
    VOL_GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    VOL_GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    VOL_GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    VOL_GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    VOL_GL_CHECK(glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        width_,
        height_,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr));
    VOL_GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
}

void GlDisplay::drawPbo(GLuint pbo, int width, int height)
{
    if (pbo == 0 || width <= 0 || height <= 0) {
        return;
    }

    resize(width, height);

    VOL_GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    VOL_GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));
    VOL_GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_));
    VOL_GL_CHECK(glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        width_,
        height_,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr));
    VOL_GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

    VOL_GL_CHECK(glUseProgram(program_));
    VOL_GL_CHECK(glActiveTexture(GL_TEXTURE0));
    VOL_GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_));
    const GLint image_location = glGetUniformLocation(program_, "u_image");
    if (image_location >= 0) {
        VOL_GL_CHECK(glUniform1i(image_location, 0));
    }

    VOL_GL_CHECK(glBindVertexArray(vao_));
    VOL_GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 3));
    VOL_GL_CHECK(glBindVertexArray(0));
    VOL_GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
    VOL_GL_CHECK(glUseProgram(0));
}

void GlDisplay::destroy()
{
    if (texture_ != 0) {
        VOL_GL_CHECK(glDeleteTextures(1, &texture_));
        texture_ = 0;
    }

    if (vao_ != 0) {
        VOL_GL_CHECK(glDeleteVertexArrays(1, &vao_));
        vao_ = 0;
    }

    if (program_ != 0) {
        VOL_GL_CHECK(glDeleteProgram(program_));
        program_ = 0;
    }

    width_ = 0;
    height_ = 0;
}

void GlDisplay::ensurePipeline()
{
    if (program_ == 0) {
        program_ = createProgram();
    }
    if (vao_ == 0) {
        VOL_GL_CHECK(glGenVertexArrays(1, &vao_));
    }
}

void GlDisplay::destroyNoThrow() noexcept
{
    try {
        destroy();
    } catch (const std::exception& exception) {
        std::cerr << "Failed to destroy OpenGL display resources cleanly: " << exception.what() << '\n';
    }
}

GLuint GlDisplay::compileShader(GLenum type, const char* source) const
{
    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
        throw std::runtime_error("Failed to create OpenGL shader object");
    }

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        const std::string log = shaderInfoLog(shader);
        glDeleteShader(shader);
        throw std::runtime_error("OpenGL shader compile failed: " + log);
    }

    return shader;
}

GLuint GlDisplay::createProgram() const
{
    const GLuint vertex_shader = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fragment_shader = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    const GLuint program = glCreateProgram();
    if (program == 0) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        throw std::runtime_error("Failed to create OpenGL shader program");
    }

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        const std::string log = programInfoLog(program);
        glDeleteProgram(program);
        throw std::runtime_error("OpenGL shader link failed: " + log);
    }

    return program;
}

} // namespace vollenia
