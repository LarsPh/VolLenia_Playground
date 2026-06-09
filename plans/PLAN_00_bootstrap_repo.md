# PLAN 00 — Bootstrap repo

## 目标

在 `D:\projects\VolLenia_Playground` 中创建一个可构建、可运行的 C++/CUDA/OpenGL 项目骨架。

这一阶段只追求：

```text
窗口能打开
ImGui 面板能显示
CUDA runtime 能查询 device
CMake 能 configure/build
```

不要做 volume renderer，不要接 cuFFT，不要接 Lenia。

## 前置条件

手动确认：

```powershell
nvidia-smi
nvcc --version
cmake --version
where cl
```

如果 `where cl` 不通过，从 Visual Studio x64 Native Tools shell 启动。

## 目标文件

建议创建：

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

依赖可以先用以下任一种方式：

```text
A. vcpkg: glfw3, glad, imgui, nlohmann-json
B. CMake FetchContent: glfw, imgui, nlohmann-json, glad/glad2
C. 手动 vendor: glfw/imgui/glad source
```

建议让 Codex 优先选择最少摩擦的方式，但要在 README 中写清楚。

## 约束

```text
1. 不引入 Falcor/OptiX/pbrt。
2. 不复制 CUDA sample 的大量代码。
3. 不实现 volume renderer。
4. 不实现 Lenia simulation。
5. 必须保持 Windows + CUDA + Visual Studio 友好。
6. 所有 CUDA API 必须有 error check helper。
```

## CMake 要求

```text
enable_language(CXX CUDA)
C++20
CUDA separable compilation 后期可开，第一阶段不强制
find_package(CUDAToolkit REQUIRED)
link CUDA::cudart
link OpenGL
link GLFW/GLAD/ImGui
```

建议支持：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release
.\build\Release\VolLenia_Playground.exe
```

如果 `native` 不可用，README 中说明如何手动指定 CUDA architecture。

## App MVP

运行后窗口显示：

```text
VolLenia Playground
FPS / frame time
CUDA device name
CUDA runtime version
按钮: Quit
```

可以先用 OpenGL 清屏颜色做背景。

## 验收标准

```text
1. Clean configure 成功。
2. Clean build 成功。
3. 运行后出现窗口和 ImGui 面板。
4. 面板显示 CUDA device 信息。
5. README 包含 build/run commands。
6. git status 干净后提交。
```

## 建议 commit

```powershell
git add .
git commit -m "build: bootstrap cuda opengl imgui app"
```
