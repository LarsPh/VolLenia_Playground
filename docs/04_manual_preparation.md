# 04 — 手动准备清单

## 必需

```text
1. NVIDIA GPU + 对应驱动
2. CUDA Toolkit 13.x 或你本机已安装版本
3. Visual Studio 2022 / Build Tools with C++ desktop workload
4. CMake
5. Git
6. 本地参考 repo:
   D:\projects\cuda_samples_13.3
   D:\projects\Lenia3D
```

## 建议安装

```text
1. NVIDIA Nsight Systems
2. NVIDIA Nsight Compute
3. RenderDoc
4. Python 3.11+，用于后期脚本/参数批处理
5. ffmpeg，后期导出视频
```

## 文献/网页建议收藏

建议你把下面资料单独放到：

```text
D:\projects\VolLenia_Playground\references\
```

候选：

```text
Lenia — Biology of Artificial Life
Lenia and Expanded Universe
SmoothLife
Flow Lenia paper / project page
Growing Neural Cellular Automata
CUDA cuFFT documentation
CUDA/OpenGL interop documentation
NVIDIA cuda-samples volumeRender source
```

这些不是第一阶段必须，但后面写项目说明和设计分支时会有用。

## 本地参考代码建议只读，不直接混入

### CUDA samples

可以从 `volumeRender` 复制/改写：

```text
ray-box intersection
front-to-back volume compositing
CUDA texture object setup
OpenGL PBO interop pattern
```

如果复制了源代码片段，保留 NVIDIA sample 原 license/copyright header。

### Lenia3D

可以参考：

```text
src/core/LeniaEngine.js
src/core/KernelGen.js
src/core/FftUtils.js
src/data/animals3D.js
```

不要直接移植 TF.js 代码结构。我们的目标是 CUDA/cuFFT 原生实现。

## 目录中的大文件策略

后续可能产生：

```text
frame dumps
video exports
large volume snapshots
parameter search logs
```

建议：

```text
outputs/
recordings/
cache/
```

默认进 `.gitignore`。真正要保存的精选结果再手动移到 `assets/examples/` 或外部 release。
