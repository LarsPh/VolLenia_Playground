# 05 — 参考代码说明

## Katielocks/Lenia3D 参考点

本项目不以 Lenia3D 的 WebGL/TF.js 代码为工程基础，但可以参考其算法结构。

### 参考文件

```text
D:\projects\Lenia3D\src\core\LeniaEngine.js
D:\projects\Lenia3D\src\core\KernelGen.js
D:\projects\Lenia3D\src\core\FftUtils.js
D:\projects\Lenia3D\src\render\Renderer3D.js
D:\projects\Lenia3D\src\data\animals3D.js
```

### 算法结构

LeniaEngine 的核心 step 可以概括为：

```text
grid -> FFT
gridFFT * kernelFFT -> potentialFFT
IFFT -> U
growth(U, μ, σ)
grid += dt * growth
grid = clip(grid, 0, 1)
```

这正是我们第一版 `LeniaRule` 的基础。

### KernelGen 的核心结构

可以概括为：

```text
create coordinate grids
create distance field
apply radial shell weights b
normalize kernel sum
roll kernel center to origin
FFT kernel
```

CUDA 版应该实现同样的数学，但不需要照搬 TF.js 的 tensor/transposition 结构。

### Renderer3D 的限制

Lenia3D 的 renderer 主要用于 Web demo。其 ray traversal 逻辑更像：

```text
沿 ray 走 voxel
遇到 density > 0 就返回颜色
```

这不是本项目想要的 emission-absorption volume rendering。我们会用它观察 UI/交互风格，但不作为 renderer 代码基础。

## NVIDIA cuda-samples volumeRender 参考点

搜索以下文件：

```text
volumeRender.cpp
volumeRender_kernel.cu
volumeRender.h
```

这个 sample 的价值在于：

```text
1. CUDA 3D texture object volume sampling
2. 1D transfer function texture
3. front-to-back alpha compositing
4. CUDA writes OpenGL PBO
5. OpenGL displays the PBO as texture
```

需要现代化的地方：

```text
FreeGLUT -> GLFW
fixed-function OpenGL -> fullscreen triangle shader
global variables -> classes
uchar volume -> float volume
hard-coded params -> ImGui/JSON
single sample app -> reusable renderer module
```

## 版权/许可证注意

如果你直接复制 CUDA sample 的函数或片段：

```text
保留原版权头
在 NOTICE 或 THIRD_PARTY_NOTICES.md 中记录来源
```

如果只是重写相同思想，则仍建议在 docs 中记录参考来源。
