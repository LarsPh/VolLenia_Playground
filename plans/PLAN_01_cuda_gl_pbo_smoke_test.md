# PLAN 01 — CUDA/OpenGL PBO smoke test

## 目标

验证 CUDA kernel 可以写入 OpenGL PBO，并通过 OpenGL fullscreen draw 显示。

这一阶段**不渲染 volume**，只渲染一个 CUDA 生成的动态图案。它的核心价值是把后续所有 renderer 都需要的这条数据链跑通：

```text
CUDA kernel -> OpenGL PBO -> GL texture -> fullscreen triangle -> ImGui overlay
```

## 前置状态

PLAN 00 已经完成：

```text
CMake + C++20 + CUDA
GLFW window
OpenGL 3.3 core context
glad
Dear ImGui
CUDA device info panel
configs/app.default.json
```

## 目标文件

建议创建/修改：

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

## 建议架构

### `CudaPbo`

职责：拥有 OpenGL PBO 和 CUDA graphics resource。

```text
- create(width, height)
- resize(width, height)
- map() -> uchar4* + byte count
- unmap()
- destroy()
```

要求：

```text
1. 使用 glGenBuffers / glBindBuffer(GL_PIXEL_UNPACK_BUFFER) / glBufferData。
2. 使用 cudaGraphicsGLRegisterBuffer 注册 PBO。
3. 注册 flags 使用 cudaGraphicsRegisterFlagsWriteDiscard。
4. 每帧 map/unmap，不读回 CPU。
5. resize 时先 unregister/delete 旧资源，再创建新资源。
6. RAII；不可复制；可移动可不做。
7. 析构必须在 OpenGL context 销毁前发生。
```

### `GlDisplay`

职责：把 PBO 内容显示到窗口。

```text
- own GL texture
- own fullscreen triangle VAO
- own shader program
- upload PBO to texture with glTexSubImage2D
- draw fullscreen triangle
```

要求：

```text
1. OpenGL core profile 下必须创建 VAO。
2. 不用 glDrawPixels。
3. 不用 fixed-function pipeline。
4. texture internal format 使用 GL_RGBA8。
5. upload format/type 使用 GL_RGBA / GL_UNSIGNED_BYTE。
6. 使用 glPixelStorei(GL_UNPACK_ALIGNMENT, 1)。
7. shader compile/link 错误必须打印清楚。
```

### `PboSmokeTest.cu`

职责：只负责生成测试图案。

```text
launchPboSmokeTest(uchar4* output, int width, int height, float time_seconds)
```

建议图案：

```text
x/y gradient
随时间移动的圆环或波纹
alpha 固定 255
```

这样肉眼能确认每帧内容来自 CUDA，而不是静态图片。

## App 集成建议

现有 framebuffer callback 可以继续只做 `glViewport`。PLAN 01 更推荐在 main loop 里轮询 framebuffer size：

```text
glfwGetFramebufferSize(window_, &width, &height)
if width/height changed:
    resize CudaPbo and GlDisplay texture
```

理由：renderer 资源属于 `App`，main loop 轮询能避免 callback 捕获复杂状态。

建议 App 持有：

```text
std::unique_ptr<CudaPbo> pbo_;
std::unique_ptr<GlDisplay> display_;
bool enable_pbo_smoke_test_ = true;
int framebuffer_width_ = 0;
int framebuffer_height_ = 0;
float animation_time_seconds_ = 0.0f;
```

每帧顺序建议：

```text
1. glfwPollEvents()
2. update framebuffer size and resize resources if needed
3. update animation time
4. glClear
5. if enable smoke test and framebuffer is non-zero:
     map PBO
     launch CUDA smoke kernel
     unmap PBO
     display PBO as fullscreen texture
6. start/build/render ImGui overlay
7. glfwSwapBuffers()
```

如果窗口 minimized 导致 framebuffer 为 0，跳过 CUDA launch 和 draw。

## CUDA/OpenGL interop 细节

需要 include：

```cpp
#include <cuda_gl_interop.h>
```

PBO 注册：

```cpp
cudaGraphicsGLRegisterBuffer(
    &resource,
    pbo,
    cudaGraphicsRegisterFlagsWriteDiscard);
```

每帧：

```cpp
cudaGraphicsMapResources(1, &resource, 0);
cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&device_ptr), &byte_count, resource);
launchPboSmokeTest(device_ptr, width, height, time_seconds);
cudaGraphicsUnmapResources(1, &resource, 0);
```

kernel launch 后检查：

```cpp
VOL_CUDA_CHECK(cudaGetLastError());
```

如果选择同步检查 Debug 版，可以只在 Debug 下做：

```cpp
VOL_CUDA_CHECK(cudaDeviceSynchronize());
```

但不要在 Release 每帧强制同步，除非为调试需要。

## CMake 要求

在 `VolLenia_Playground` target 中加入新源文件：

```text
src/render/GlDisplay.cpp
src/render/CudaPbo.cpp
src/render/PboSmokeTest.cu
```

确保仍然链接：

```text
CUDA::cudart
OpenGL::GL
glfw
glad_gl_core_33
imgui_backend
```

一般不需要为这个阶段启用 `CUDA_SEPARABLE_COMPILATION`。

## UI

ImGui 增加一个 section：

```text
CUDA/OpenGL interop
  checkbox: Enable PBO smoke test
  framebuffer size
  PBO byte size
  animation time
  status / last error
```

UI 不需要做复杂参数编辑；只要能确认资源状态即可。

## 约束

```text
1. 不做 volume texture。
2. 不接 cuFFT。
3. 不接 Lenia。
4. 不使用 glDrawPixels。
5. 不使用 OpenGL fixed-function pipeline。
6. 不把 PBO 内容读回 CPU。
7. 不引入 Falcor、OptiX、pbrt、Vulkan、DirectX。
8. 不把 smoke-test kernel 和未来 volume renderer 写死耦合。
```

## 重点验收标准

```text
1. 运行后看到 CUDA 生成的动态图案。
2. ImGui overlay 仍正常显示。
3. resize 后图案比例/尺寸正确。
4. minimize/restore 不崩溃。
5. Debug/Release 都能跑。
6. 关闭窗口时无 CUDA resource leak / crash。
7. 所有 CUDA map/unmap/register/kernel launch 错误都有检查。
8. 没有 CPU readback。
```

## 建议手动测试

```powershell
cmake --build --preset release
.\build\Release\VolLenia_Playground.exe
```

如果 `native` CUDA arch 有问题：

```powershell
cmake --preset vs2022-x64-sm89
cmake --build --preset release-sm89
.\build-sm89\Release\VolLenia_Playground.exe
```

测试动作：

```text
1. 观察动态图案是否随时间变化。
2. 拖动 resize 窗口。
3. 最小化再恢复。
4. 切换 Enable PBO smoke test。
5. 关闭窗口。
```

## 建议 commit

```powershell
git add .
git commit -m "interop: add cuda opengl pbo smoke test"
```
