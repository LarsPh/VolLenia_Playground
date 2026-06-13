# Review — PLAN 00 bootstrap result

Repo reviewed: `LarsPh/VolLenia_Playground`

## Verdict

Proceed to PLAN 01. The bootstrap is healthy for the intended next step.

The current repo has the important pieces in place:

- CMake project is C++20 + CUDA, and already links CUDA runtime, OpenGL, GLFW, glad, ImGui, and nlohmann/json.
- The app shell initializes GLFW, creates an OpenGL 3.3 core context, loads glad, initializes ImGui, queries CUDA device info, and renders a status panel.
- Error-checking helpers for CUDA and OpenGL already exist.
- Config loading and CMake presets are good enough for the next milestone.

## Minor risks to handle in PLAN 01

These are not blockers, but the PBO interop plan should be more explicit about them:

1. **Destroy CUDA/GL resources before destroying the GL context.**  
   PBO unregister/delete must happen before `glfwDestroyWindow()` / `glfwTerminate()`.

2. **Do framebuffer resize from App-owned state.**  
   The current framebuffer callback only calls `glViewport`. PLAN 01 should either use `glfwSetWindowUserPointer` or, simpler, poll framebuffer size in the main loop and call renderer resize when it changes.

3. **OpenGL core profile requires a VAO for fullscreen draw.**  
   The fullscreen triangle path should create and bind a VAO. Do not use fixed-function pipeline.

4. **Register OpenGL PBO with CUDA only after GL context creation.**  
   Use `<cuda_gl_interop.h>` and `cudaGraphicsGLRegisterBuffer(..., cudaGraphicsRegisterFlagsWriteDiscard)`.

5. **Keep the smoke-test generator separate from display.**  
   `CudaPbo` and `GlDisplay` should be reusable by the later volume renderer. `PboSmokeTest.cu` should be easy to delete/replace in PLAN 02.

## Recommended next action

Apply the updated PLAN 01 / CODEX 01 files in this bundle, then run Codex on:

```text
codex_prompts/CODEX_01_cuda_gl_pbo_smoke_test.md
```
