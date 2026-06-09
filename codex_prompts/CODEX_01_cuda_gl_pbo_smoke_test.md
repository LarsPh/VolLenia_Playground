# Codex prompt — 01 CUDA/OpenGL PBO smoke test

You are working in `D:\projects\VolLenia_Playground`.

The repo should already have a C++20/CUDA/OpenGL/GLFW/ImGui skeleton from PLAN 00.

## Goal

Add a CUDA/OpenGL PBO interop smoke test. CUDA should write an animated `uchar4` image into an OpenGL PBO, and OpenGL should display it fullscreen.

## Constraints

- Do not implement volume rendering yet.
- Do not use cuFFT yet.
- Do not implement Lenia yet.
- Do not use `glDrawPixels` or fixed-function OpenGL.
- Do not read PBO data back to CPU.
- Do not introduce Falcor, OptiX, pbrt, Vulkan, or DirectX.

## Suggested files

```text
src/render/GlDisplay.h
src/render/GlDisplay.cpp
src/render/CudaPbo.h
src/render/CudaPbo.cpp
src/render/PboSmokeTest.h
src/render/PboSmokeTest.cu
src/core/CudaCheck.h
src/core/GlCheck.h
src/app/App.cpp
CMakeLists.txt
```

## Implementation details

- Create an OpenGL PBO sized to framebuffer width * height * sizeof(uchar4).
- Register it with CUDA using `cudaGraphicsGLRegisterBuffer`.
- Each frame:
  - map CUDA graphics resource
  - get mapped pointer
  - launch CUDA kernel writing a moving gradient/ring pattern
  - unmap resource
  - use OpenGL to display the PBO through a texture/fullscreen triangle
- Recreate PBO and texture on window resize.
- Add robust cleanup on shutdown.

## UI

Add an ImGui section:

- interop status
- framebuffer size
- animation time
- checkbox: enable CUDA PBO smoke test

## Acceptance criteria

- Running app displays an animated image produced by CUDA.
- Window resize works.
- No CPU readback.
- CUDA/OpenGL errors are checked.
- Debug and Release builds work.

After implementation, summarize the changes and give build/run commands.
