# Codex prompt — 01 CUDA/OpenGL PBO smoke test

You are working in:

```text
D:\projects\VolLenia_Playground
```

The repo already has a C++20/CUDA/OpenGL/GLFW/ImGui skeleton from PLAN 00. The app currently opens a GLFW OpenGL 3.3 core window, initializes glad and ImGui, queries CUDA device information, and shows a status panel.

## Goal

Add a CUDA/OpenGL PBO interop smoke test.

CUDA should write an animated `uchar4` RGBA image into an OpenGL PBO every frame. OpenGL should display that PBO fullscreen using a texture and fullscreen triangle, with the existing ImGui panel drawn as an overlay.

This milestone does **not** implement volume rendering, cuFFT, or Lenia simulation.

## Constraints

- Do not implement volume rendering yet.
- Do not use cuFFT yet.
- Do not implement Lenia yet.
- Do not use `glDrawPixels`.
- Do not use fixed-function OpenGL.
- Do not read PBO data back to CPU.
- Do not introduce Falcor, OptiX, pbrt, Vulkan, DirectX, or new large dependencies.
- Keep smoke-test image generation separate from the reusable PBO/display infrastructure.
- Keep OpenGL 3.3 core compatibility.

## Suggested files

Create/modify:

```text
src/render/GlDisplay.h
src/render/GlDisplay.cpp
src/render/CudaPbo.h
src/render/CudaPbo.cpp
src/render/PboSmokeTest.h
src/render/PboSmokeTest.cu

src/app/App.h
src/app/App.cpp
src/app/UiPanel.h
src/app/UiPanel.cpp

src/core/CudaCheck.h
src/core/GlCheck.h
CMakeLists.txt
```

## Required design

### `CudaPbo`

Create an RAII class that owns:

```text
GLuint pbo
cudaGraphicsResource* cuda_resource
int width
int height
size_t byte_size
bool mapped
```

It should support:

```text
create/resize(width, height)
map() -> uchar4* device pointer
unmap()
destroy()
```

Implementation details:

- Include `<cuda_gl_interop.h>`.
- Use `glGenBuffers`.
- Bind `GL_PIXEL_UNPACK_BUFFER`.
- Allocate `width * height * sizeof(uchar4)` using `glBufferData(..., GL_STREAM_DRAW)`.
- Register with CUDA using:

```cpp
cudaGraphicsGLRegisterBuffer(
    &resource,
    pbo,
    cudaGraphicsRegisterFlagsWriteDiscard);
```

- On resize/destruction, unregister CUDA graphics resource before deleting the GL buffer.
- Ensure cleanup happens before the GLFW/OpenGL context is destroyed.
- The class should be non-copyable.
- Check all CUDA and GL errors.

### `GlDisplay`

Create a reusable OpenGL display helper that owns:

```text
GLuint texture
GLuint vao
GLuint shader program
```

It should:

- Create an RGBA8 texture matching framebuffer size.
- Create a fullscreen triangle pipeline.
- Create and bind a VAO, because OpenGL core profile requires it.
- Copy the PBO into the texture using:

```cpp
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
glBindTexture(GL_TEXTURE_2D, texture);
glTexSubImage2D(..., GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
```

- Draw a fullscreen triangle with a simple shader.
- Print clear shader compile/link errors.
- Use `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)` before upload.

### `PboSmokeTest.cu`

Add a host wrapper such as:

```cpp
void launchPboSmokeTest(uchar4* output, int width, int height, float time_seconds);
```

The CUDA kernel should write:

```text
x/y gradient
animated ring or wave pattern
alpha = 255
```

Use block size like `16x16`. Check kernel launch errors with `VOL_CUDA_CHECK(cudaGetLastError())`.

In Debug builds, it is acceptable to optionally call `cudaDeviceSynchronize()` after launch for easier debugging. Avoid forced synchronization every frame in Release unless needed.

## App integration

Add members to `App` for the PBO/display/smoke-test state. Use `std::unique_ptr` if that makes cleanup ordering explicit.

Recommended members:

```cpp
std::unique_ptr<CudaPbo> pbo_;
std::unique_ptr<GlDisplay> display_;
bool enable_pbo_smoke_test_ = true;
int framebuffer_width_ = 0;
int framebuffer_height_ = 0;
float animation_time_seconds_ = 0.0f;
```

Resize strategy:

- Keep the existing framebuffer callback for `glViewport`.
- In the main loop, call `glfwGetFramebufferSize`.
- If the size changed and both dimensions are > 0, resize `CudaPbo` and `GlDisplay`.
- If either dimension is 0, skip CUDA launch and fullscreen draw for that frame.

Suggested per-frame order:

```text
1. glfwPollEvents()
2. update frame time and animation time
3. get framebuffer size and resize render resources if needed
4. glClear
5. if smoke test enabled and framebuffer size is non-zero:
     map PBO
     launch CUDA smoke kernel
     unmap PBO
     display PBO fullscreen
6. build and render ImGui
7. glfwSwapBuffers()
```

The smoke image should appear behind the ImGui panel.

## UI changes

Extend the existing ImGui panel with a section:

```text
CUDA/OpenGL interop
  checkbox: Enable PBO smoke test
  framebuffer size
  PBO byte size
  animation time
  status or last error
```

The checkbox should actually disable/enable the smoke draw path.

## CMake changes

Add new render sources to the existing executable target:

```text
src/render/GlDisplay.cpp
src/render/CudaPbo.cpp
src/render/PboSmokeTest.cu
```

Keep linking:

```text
CUDA::cudart
OpenGL::GL
glfw
glad_gl_core_33
imgui_backend
nlohmann_json::nlohmann_json
```

Do not enable unnecessary CUDA separable compilation for this milestone.

## Acceptance criteria

- Running the app displays an animated image produced by CUDA.
- Existing ImGui overlay remains visible.
- Window resize works.
- Minimize/restore does not crash.
- The checkbox can disable/enable the smoke pattern.
- No CPU readback is used.
- CUDA/OpenGL errors are checked.
- Debug and Release builds work.
- Shutdown does not crash and does not leave mapped/unregistered resources.

## Build/run commands

Use the existing preset:

```powershell
cmake --build --preset release
.\build\Release\VolLenia_Playground.exe
```

If CUDA native architecture configuration fails, use the sm_89 preset:

```powershell
cmake --preset vs2022-x64-sm89
cmake --build --preset release-sm89
.\build-sm89\Release\VolLenia_Playground.exe
```

After implementation, summarize the changed files, how the resource lifetime is handled, and the exact build/run command that passed locally.
