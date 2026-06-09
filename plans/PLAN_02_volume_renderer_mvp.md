# PLAN 02 — CUDA volume renderer MVP

## 目标

实现最小 CUDA volume renderer：渲染 synthetic 3D volume，不接 Lenia simulation。

参考 `cuda_samples_13.3` 的 `volumeRender` 思路，但使用当前项目架构重写。

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

如果直接复制代码片段，保留 license header。

## 目标文件

建议创建/修改：

```text
src/render/CudaVolumeRenderer.h
src/render/CudaVolumeRenderer.cu
src/render/RenderParams.h
src/render/TransferFunction.h
src/render/TransferFunction.cpp
src/render/SyntheticVolume.h
src/render/SyntheticVolume.cu
src/app/Camera.h
src/app/Camera.cpp
src/app/UiPanel.cpp
CMakeLists.txt
```

## 数据结构

第一版：

```text
synthetic volume: float32 [nx, ny, nz] linear CUDA buffer
render texture: cudaArray 3D, float channel
texture object: normalized coords, linear filtering, clamp address
output: existing OpenGL PBO uchar4
```

建议默认：

```text
volume size: 128^3
render size: window framebuffer
```

## Synthetic volume presets

至少实现：

```text
Sphere density
Shell density
Several Gaussian blobs
Noise blob 可选
```

这样先调 renderer，不受 Lenia 影响。

## Ray marching MVP

必须包含：

```text
ray-box intersection for unit cube or [-1,1]^3 box
fixed step size
sample 3D texture
threshold
Beer-Lambert alpha: alpha = 1 - exp(-density_scale * density * step_size)
front-to-back compositing
transmittance early exit
brightness/exposure
```

## UI 参数

```text
volume preset
step_size
density_scale
threshold
brightness
max_steps
render mode: emission_absorption first
camera orbit controls
reset camera
```

## 约束

```text
1. 不接 Lenia。
2. 不接 cuFFT。
3. 不做 gradient shading，除非非常简单。
4. 不做 empty-space skipping。
5. 不引入 OptiX/Falcor/pbrt。
```

## 验收标准

```text
1. 能看到半透明 sphere/shell/blob。
2. ImGui 参数实时影响渲染。
3. 相机可以 orbit/zoom/pan。
4. 无 CPU readback。
5. resize 正常。
```

## 建议 commit

```powershell
git add .
git commit -m "render: add cuda volume ray marcher mvp"
```
