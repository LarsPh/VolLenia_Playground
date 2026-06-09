# PLAN 03 — Single-channel 3D Lenia simulation MVP

## 目标

接入 cuFFT，完成 single-channel 3D Lenia simulation，并把结果交给 volume renderer 显示。

## 参考文件

Lenia3D 本地参考：

```text
D:\projects\Lenia3D\src\core\LeniaEngine.js
D:\projects\Lenia3D\src\core\KernelGen.js
D:\projects\Lenia3D\src\core\FftUtils.js
```

核心等价逻辑：

```text
grid FFT
multiply by kernel FFT
inverse FFT to potential U
growth(U, μ, σ)
A += dt * growth
clip A to [0,1]
```

## 目标文件

建议创建/修改：

```text
src/core/VolumeDesc.h
src/sim/DeviceVolume.h
src/sim/DeviceVolume.cu
src/sim/Rule.h
src/sim/LeniaRule.h
src/sim/LeniaRule.cu
src/sim/CufftContext.h
src/sim/CufftContext.cpp
src/sim/KernelGenerator.h
src/sim/KernelGenerator.cu
src/sim/GrowthFunctions.h
src/sim/GrowthFunctions.cu
src/render/RenderVolumePacker.h
src/render/RenderVolumePacker.cu
configs/lenia.default.json
src/app/UiPanel.cpp
CMakeLists.txt
```

## FFT plan

建议使用 R2C/C2R：

```text
A:     float [nx, ny, nz]
Ahat:  cufftComplex [nx, ny, nz/2+1]
Khat:  cufftComplex [nx, ny, nz/2+1]
Uhat:  cufftComplex [nx, ny, nz/2+1]
U:     float [nx, ny, nz]
```

每次 inverse FFT 后要乘以：

```text
1.0f / (nx * ny * nz)
```

否则 potential 会按体素数放大。

## Kernel generator MVP

先实现 Lenia3D-style radial shell kernel：

```text
create distance field around center
r = distance / radius
if r < 1:
  shell index = floor(r * B)
  local_r = fract(r * B)
  core(local_r) * shell_weight[index]
else:
  0
normalize sum to 1
roll center to origin for circular convolution
FFT -> Khat
```

第一版 core 函数至少实现：

```text
polynomial bump: (4r(1-r))^4
Gaussian/bump-like core later
step core optional
```

## Growth function MVP

至少实现：

```text
Gaussian growth:
G(U) = 2 * exp(-(U - μ)^2 / (2σ^2)) - 1
```

Update：

```text
A = clamp(A + dt * G(U), 0, 1)
```

## Initial state MVP

至少实现：

```text
random blob centered in volume
sphere/shell seed
reset with seed number
```

不要一开始导入 animals3D RLE。

## Renderer integration

新增 `RenderVolumePacker`：

```text
input: DeviceVolume A, selected channel 0
output: render cudaArray / texture update
```

renderer 不应直接知道 LeniaRule。

## UI 参数

```text
Play/Pause
Single step
Reset
steps_per_frame
volume size display
μ
σ
dt or T
kernel radius
shell weights preset
random seed
mass / min / max debug stats 可选
```

## 验收标准

```text
1. 128^3 single-channel Lenia 可以连续 step。
2. renderer 显示当前 A。
3. Play/Pause/Step/Reset 可用。
4. 参数改动后 kernel 可重建。
5. 没有 CPU readback 大 volume。
6. 单步后没有 NaN/Inf。
```

## 建议 commit

```powershell
git add .
git commit -m "sim: add single channel 3d lenia mvp"
```
