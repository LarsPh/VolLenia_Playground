# Codex prompt — 00 Bootstrap repo

You are working in `D:\projects\VolLenia_Playground` on Windows.

Build a minimal C++20/CUDA/OpenGL project skeleton for **VolLenia Playground**.

## Context

The long-term project is a CUDA/cuFFT 3D Lenia simulation with a CUDA volume ray marcher and OpenGL display. For this task, do **not** implement volume rendering or Lenia simulation yet.

## Goal

Create a buildable app that opens a GLFW/OpenGL window, initializes Dear ImGui, and displays CUDA device information.

## Constraints

- Do not introduce Falcor, OptiX, pbrt, Vulkan, or DirectX.
- Do not copy CUDA samples code yet.
- Keep dependencies minimal and explain the chosen dependency method in README.
- Use CMake.
- Use C++20.
- Enable CUDA language and link CUDA runtime.
- Add CUDA error-check helper.
- App must build and run on Windows with Visual Studio + CUDA.

## Create/modify files

Suggested files:

```text
.gitignore
README.md
CMakeLists.txt
CMakePresets.json
configs/app.default.json
src/main.cpp
src/app/App.h
src/app/App.cpp
src/app/Camera.h
src/app/Camera.cpp
src/app/UiPanel.h
src/app/UiPanel.cpp
src/core/CudaCheck.h
src/core/GlCheck.h
src/core/Version.h.in
third_party/README.md
```

## App behavior

On launch:

- open a window titled `VolLenia Playground`
- render an ImGui panel
- show FPS/frame time
- show CUDA device name and CUDA runtime version
- provide a Quit button

## Build commands to document

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release
.\build\Release\VolLenia_Playground.exe
```

If `native` CUDA architecture is not supported by the installed CMake/CUDA combination, document how to set it manually.

## Acceptance criteria

- Clean configure succeeds.
- Clean build succeeds.
- Executable opens a window.
- ImGui panel shows CUDA device info.
- README contains dependency/build/run instructions.
- No volume renderer or Lenia code yet.

After implementation, summarize files changed and exact commands to build/run.
