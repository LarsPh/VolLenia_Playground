# 00 — 项目背景、目标与边界

## 背景

这个项目从三个观察出发：

1. **3D Lenia 的主状态天然是 volume field。** 也就是每个空间点/voxel 有一个连续 density 或多通道 state，而不是只有表面。
2. **漂亮渲染更适合 volume rendering。** Lenia 的状态像活性密度、组织、雾、半透明生物体；直接把它当作 volume cloud / participating medium 的简化模型，比 SDF 表面更能保留内部结构。
3. **未来实验会越来越复杂。** 后续可能接 multi-channel、environment/resource、spectral-logit field、learnable kernel / growth、Flow-Lenia-like transport、甚至 neural update rule。因此第一阶段架构要避免写死 single scalar toy demo。

## 第一阶段目标

第一阶段目标是做一个可交互的 **Volumetric 3D Lenia Playground**：

```text
C++/CUDA app
CUDA + cuFFT simulation
CUDA volume ray marcher
OpenGL display
GLFW + Dear ImGui UI
JSON config
可保存截图/参数
```

最小可用版本必须能做到：

```text
1. 启动窗口和 ImGui 面板。
2. 用 CUDA 写 OpenGL PBO，并显示 fullscreen 图像。
3. 用 CUDA ray marcher 渲染一个 synthetic 3D volume。
4. 接入 cuFFT，跑 single-channel 3D Lenia step。
5. renderer 直接渲染当前 Lenia state。
6. 参数可以实时调：dt、growth μ/σ、kernel radius、density scale、step size、transfer function。
```

## 非目标

第一阶段明确不做：

```text
1. 不直接上 pbrt-v4 wavefront renderer。
2. 不直接上 Falcor render graph。
3. 不直接上 OptiX path tracing。
4. 不做完整 physically based multiple scattering。
5. 不做复杂 RL / neural training。
6. 不做 Gaussian Splat state。
7. 不做 sparse NanoVDB volume。
```

这些都可以作为后期 backend 或研究分支，但不应该拖慢第一阶段。

## 为什么先走 CUDA ray marcher + OpenGL PBO

原因很直接：

```text
simulation 已经在 CUDA/cuFFT
rendering 也在 CUDA 中采样 volume
OpenGL 只负责显示最终 image texture
无 CPU readback
最少 interop 心智负担
最容易调试 GPU state
```

第一阶段 OpenGL 不是主要 renderer，而是窗口和显示后端。

## 参考项目的角色

### `cuda_samples_13.3`

参考 NVIDIA CUDA samples 的 `volumeRender`：

```text
可参考：
  ray-box intersection
  CUDA texture object / 3D texture sampling
  transfer function texture
  front-to-back alpha compositing
  CUDA/OpenGL PBO interop

不要照搬：
  FreeGLUT app structure
  fixed-function OpenGL quad
  uchar-only volume assumption
  sample-level UI style
```

如果复制代码片段，必须保留原 license/copyright header。

### `Lenia3D`

参考 Katielocks/Lenia3D：

```text
可参考：
  3D Lenia rule structure
  FFT convolution + kernel spectrum + growth + clip
  radial shell kernel generation
  predefined animals/pattern params
  UI 参数命名

不要作为：
  CUDA simulation 基础
  高性能 3D FFT 基础
  final volume rendering 基础
```

Lenia3D 的 WebGL renderer 更像 first-hit occupancy preview，不是本项目第一阶段要做的 alpha-compositing volume renderer。

## 长期愿景

第一阶段完成后，项目可以向三个方向发展：

```text
A. 视觉主线：更好的 volume renderer / lighting / export / cinematic backend
B. Lenia 主线：kernel zoo / multi-channel / environment / Flow-like dynamics
C. Field 主线：spectral-logit Lenia / function-space state / projection dynamics
D. Learning 主线：learnable kernels / growth MLP / quality diversity / 2D ecology lab
```

建议主线仍以 3D renderer playground 为中心，复杂学习实验先在 2D 并行验证。
