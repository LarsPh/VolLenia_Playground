# PLAN 02 — CUDA volume renderer MVP

## 目标

实现最小 CUDA volume renderer：渲染 synthetic 3D volume，不接 Lenia simulation。

这一阶段的核心不是追求最终画质，而是跑通下一条 GPU 数据链：

```text
GPU volume buffer -> CUDA 3D array / texture object -> CUDA ray marcher -> existing OpenGL PBO -> GL fullscreen display
```

参考 `cuda_samples_13.3` 的 `volumeRender` 思路，但使用当前项目架构重写。继续保留 Plan 1 的 PBO/GlDisplay 输出路径。

## 参考文件

在本地 CUDA samples 中搜索：

```powershell
Get-ChildItem -Path D:\projects\cuda_samples_13.3 -Recurse -Filter volumeRender_kernel.cu
Get-ChildItem -Path D:\projects\cuda_samples_13.3 -Recurse -Filter volumeRender.cpp
```

可参考：

```text
ray-box intersection
tex3D<float>() sampling
front-to-back alpha compositing
opacity early exit
transfer function idea
```

如果直接复制代码片段，保留 license header。更推荐只借鉴结构并重写。

## 目标文件

建议创建/修改：

```text
src/render/CudaVolumeRenderer.h
src/render/CudaVolumeRenderer.cu
src/render/RenderParams.h
src/render/SyntheticVolume.h
src/render/SyntheticVolume.cu
src/app/Camera.h
src/app/Camera.cpp
src/app/UiPanel.cpp
src/app/App.h
src/app/App.cpp
CMakeLists.txt
configs/app.default.json
```

`TransferFunction.*` 可以暂缓。第一版可以在 CUDA kernel 里用 2-4 个简单参数生成颜色；等 renderer 跑通后再把 transfer function 抽成独立模块。

## 数据结构

第一版：

```text
synthetic volume: float32 [nx, ny, nz] linear CUDA buffer
render texture: cudaArray 3D, one float channel
texture object: normalized coords, linear filtering, clamp address
output: existing OpenGL PBO uchar4
```

建议默认：

```text
volume size: 128^3
render size: window framebuffer
box: [-1, 1]^3 in world/object space
camera: orbit around origin
```

后续接 Lenia 时，simulation 仍然保留 linear CUDA buffer / cuFFT buffers；renderer 每帧或按需把 selected render channel copy/pack 到 CUDA 3D array。不要为了 renderer 把 simulation 主状态改成 cudaArray。

## Synthetic volume presets

至少实现：

```text
Sphere density
Shell density
Gaussian blobs
Lenia-like phantom: shell + internal blobs + wispy bands
```

可选 debug presets：

```text
Axis ramps: x/y/z gradient，用来检查坐标方向
Checker/stripes: 用来检查 filtering 和 step size
Noise blob: 用来检查 alpha accumulation
```

这些 scene 的目的不是好看，而是帮助判断：

```text
ray-box intersection 是否正确
camera orientation 是否正确
3D texture coords 是否正确
alpha compositing 是否正确
threshold/density/step size 是否直观
```

## Ray marching MVP

必须包含：

```text
ray-box intersection for [-1, 1]^3 box
perspective camera ray generation
fixed step size
sample 3D texture with tex3D<float>()
threshold
Beer-Lambert alpha: alpha = 1 - exp(-density_scale * density * step_size)
front-to-back compositing
transmittance early exit
brightness/exposure
max_steps guard
```

建议 render modes：

```text
EmissionAbsorption: 主模式
MIP: max intensity projection，debug 用
FirstHit/Threshold: iso-surface-ish debug，用来检查边界和相机
```

MIP / FirstHit 不是最终画质要求，但非常适合调试 volume sampling。

## Camera MVP

Plan 2 需要真正的交互相机。建议先做：

```text
Left drag: orbit yaw/pitch
Right drag or middle drag: pan，可选
Mouse wheel: zoom/distance
Reset camera
```

如果 pan 暂时麻烦，至少实现 orbit + zoom + reset。

Camera 输出给 CUDA renderer 的参数可以是：

```text
camera_position
camera_forward
camera_right
camera_up
fov_y_degrees
aspect
```

不要在 CUDA kernel 内依赖 OpenGL matrix state。

## UI 参数

```text
volume preset
volume resolution
regenerate volume button
step_size
density_scale
threshold
brightness/exposure
max_steps
render mode
early exit threshold
camera distance / fov
reset camera
```

建议加 debug/status：

```text
volume dimensions
render target dimensions
CUDA 3D array/texture status
last render error
```

## 约束

```text
1. 不接 Lenia。
2. 不接 cuFFT。
3. 不做 empty-space skipping。
4. 不引入 OptiX/Falcor/pbrt/Vulkan/DirectX。
5. 不读回 CPU。
6. 不做双 PBO/ring buffer，除非单 PBO 出现明显 stall。
7. 不把 PBO 输出路径替换成 CUDA 直接写 GL texture；当前阶段继续复用 Plan 1 输出。
```

## 性能取舍

当前阶段可以忽略这些优化：

```text
uniform location cache
PBO -> GL texture copy removal
double PBO / ring buffer
empty-space skipping
adaptive ray marching
gradient lighting
```

原因：Plan 2 的主要风险是坐标、相机、texture object、alpha compositing，而不是最终性能。先写清楚 profiler/status，等 renderer 和 simulation 都接上后再优化真正的瓶颈。

但可以顺手避免明显低效：

```text
不要每帧重建 cudaArray/texture object，只有 volume resolution 或 preset 变更时重建/重新上传。
不要每帧重建 GL shader/VAO/texture。
不要在 CUDA render kernel 后做 debug cudaDeviceSynchronize，除非 Debug build 或错误诊断。
```

## 验收标准

```text
1. 能看到半透明 sphere/shell/blob/Lenia-like phantom。
2. ImGui 参数实时影响渲染。
3. 相机可以 orbit/zoom，reset 正常。
4. MIP 或 FirstHit debug mode 至少一个可用。
5. 无 CPU readback。
6. resize 正常。
7. 关闭窗口时无 CUDA graphics/cudaArray/texture object leak 或 crash。
```

## 建议 commit

```powershell
git add .
git commit -m "render: add cuda volume ray marcher mvp"
```
