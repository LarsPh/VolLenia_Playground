# PLAN 05 — Biology polish, in-place FFT, kernel visualization, and scaled cells

## 目标

在 PLAN 04 已经能导入 Lenia3D animals 的基础上，做一轮短期实用增强：

```text
1. 简化 NaN/Inf validation：去掉 GUI checkbox，改成每 frame 最多读回一次 invalid flag。
2. 优化 FFT spectrum multiply：删除 potential_spectrum_，改为 state_spectrum_ *= kernel_spectrum_。
3. 保留 spatial_kernel_，不做过度省内存重构。
4. 增加 kernel radial profile / 2D kernel visualization Python 工具。
5. 增加 imported animal cells 的 scaling/resampling path，让小 reference animals 能在更大 canvas 中以更大身体显示。
```

这一步仍然不做完整参数搜索、不做 multi-channel、不做 Flow-Lenia、不做 renderer path tracing。

---

## 背景和动机

PLAN 04 解决了 “animal = cells + params” 的耦合导入问题；但现在仍有几个短期痛点：

```text
- Validate NaN/Inf every step checkbox 不适合长期保留，逻辑和性能调试都更复杂。
- 当前 FFT convolution 多占一个 potential_spectrum_ complex buffer。
- animals 的 cells 通常很小，center padding 到 128/256 后视觉上太小，volume rendering 的优势看不出来。
- kernel 参数 b/kn/R/shell_count 很难直觉理解，需要可视化 profile。
```

这一步的目标是让 biology exploration 更舒服，而不是追求 Lenia3D bit-exact parity。

---

## 任务 A — invalid flag validation UI 简化

### 当前问题

当前 `LeniaConfig` 有：

```cpp
bool validate_nan_inf_every_step = false;
```

UI 里有：

```cpp
Validate NaN/Inf every step
```

simulation 现在根据这个 bool 决定是否在 update kernel 中检查 NaN/Inf，并且原实现倾向于每 step 读回一次 invalid flag。

### 新设计

删除 GUI checkbox 和 config 字段，默认固定采用：

```text
simulateSteps(frame_steps):
  cudaMemset(invalid_flag_, 0) 一次
  repeat frame_steps:
    R2C
    in-place multiply
    C2R
    updateStateKernel(..., invalid_flag_)
  cudaMemcpy invalid_flag_ -> CPU 一次
```

如果以后需要定位具体哪一个 step 出错，把 `steps_per_frame` 改成 1 即可。

### 修改点

```text
src/app/App.h
src/app/App.cpp
src/app/UiPanel.cpp
src/sim/LeniaSimulation.h
src/sim/LeniaSimulation.cu
configs/lenia.default.json
```

具体要求：

```text
- 删除 LeniaConfig::validate_nan_inf_every_step。
- 删除 config 中 validate_nan_inf_every_step 读取。
- 删除 ImGui checkbox。
- LeniaSimulation::simulateSteps 改成 simulateSteps(int steps)。
- updateStateKernel 不再传 validate bool；始终检查 !isfinite(next) 并 atomicExch invalid_flag。
- 每次 simulateSteps 只在整个 steps loop 开始前 cudaMemset invalid_flag_。
- 整个 steps loop 结束后 cudaMemcpy invalid_flag_ 一次。
```

验收：

```text
- UI 不再显示 Validate NaN/Inf every step。
- steps_per_frame > 1 时只发生一次 invalid flag readback。
- 出现 NaN/Inf 时仍会暂停/报告。
```

---

## 任务 B — FFT spectrum multiply 改成 in-place

### 当前路径

```text
R2C(state -> state_spectrum_)
multiplySpectrumKernel(state_spectrum_, kernel_spectrum_ -> potential_spectrum_)
C2R(potential_spectrum_ -> potential_)
```

### 新路径

```text
R2C(state -> state_spectrum_)
multiplySpectrumInPlaceKernel(state_spectrum_ *= kernel_spectrum_)
C2R(state_spectrum_ -> potential_)
```

### 修改点

```text
src/sim/LeniaSimulation.h
src/sim/LeniaSimulation.cu
```

具体要求：

```text
- 删除 cufftComplex* potential_spectrum_ 成员。
- 删除 potential_spectrum_ cudaMalloc / cudaFree。
- 把 multiplySpectrumKernel 改成 multiplySpectrumInPlaceKernel(cufftComplex* spectrum, const cufftComplex* kernel, size_t count)。
- simulateSteps 中 C2R 直接使用 state_spectrum_ 作为 input。
- 保留 potential_ float volume，不要为了省显存把它删掉。
```

验收：

```text
- 编译通过。
- 128^3 simulation 仍能连续运行。
- visual behavior 与修改前基本一致。
- GPU memory usage 少一个 packed complex spectrum buffer。
```

---

## 任务 C — 保留 spatial_kernel_

不要为了省内存删除 `spatial_kernel_` 或复用 `potential_`。理由：

```text
- 当前 kernel rebuild/debug 需要清晰的数据生命周期。
- spatial_kernel_ 只在 kernel rebuild 时使用，内存代价可接受。
- 复用 potential_ 会增加状态机复杂度，容易影响后续 profile/visualization/debug。
```

本任务没有代码改动，只要求不要在任务 B 中误删 `spatial_kernel_`。

---

## 任务 D — kernel visualization Python 工具

### 新增脚本

```text
scripts/plot_lenia_kernel_profiles.py
```

### 输入

支持：

```text
--manifest configs/lenia3d_reference/animals.json
--animal-index 0
--all
--limit 12
--size 128
--output-dir outputs/kernel_profiles
```

第一版只需要读 animal manifest 中的：

```text
R, b, kn, name/code/slug
```

### 输出

每个 animal 输出至少：

```text
<slug>_radial_profile.png
<slug>_kernel_origin_slice.png
<slug>_kernel_centered_slice.png
```

可选输出：

```text
<slug>_kernel_profile.json
```

### 计算要求

实现与 C++ 当前 kernel builder 一致的 radial shell kernel：

```text
q = wrapped_distance / R
if q < 1:
  shell_position = q * shell_count
  shell_index = floor(shell_position)
  local_r = fract(shell_position)
  value = b[shell_index] * core(local_r, kn)
else:
  value = 0
normalize kernel sum to 1
```

其中 core 至少支持：

```text
kn=1 polynomial bump: (4r(1-r))^4
kn=2 exponential bump: exp(4 - 1/(r(1-r))) for 0<r<1
kn=3 step
kn=4 staircase
```

2D slice：

```text
origin slice: z = 0, FFT-origin wrapped view
centered slice: fftshift(origin slice), human-centered view
```

脚本依赖：

```text
numpy
matplotlib
```

不要把这些依赖加进 CMake。README/脚本错误信息里提示用户自行安装即可。

---

## 任务 E — imported animal cells scaling/resampling

### 目标

让 Lenia3D reference animal 在更大 volume 中有更大的视觉体积，例如：

```text
original cells: 14^3
scale factor: 4.0
scaled cells: 56^3
canvas: 128^3 or 256^3
R: original R * 4.0 if auto-scale R enabled
```

### 新增数据结构

建议添加：

```cpp
enum class CellResampleMode {
    Nearest = 0,
    Trilinear,
};

struct ImportedCellScaleConfig {
    bool use_scaled_cells = false;
    bool auto_scale_R = true;
    bool auto_set_canvas = false; // optional; default false
    float scale = 1.0f;
    CellResampleMode mode = CellResampleMode::Trilinear;
};
```

可以放在 `App.h` 或新文件 `src/io/CellResampler.h`，按当前架构方便程度决定。

### 新增 GPU resampler

建议新增：

```text
src/io/CellResampler.h
src/io/CellResampler.cu
```

或放在 `src/sim/` 下。职责：

```text
input: DeviceVolumeView source_cells
output: DeviceVolume scaled_cells
scale: float
mode: nearest / trilinear
```

注意：

```text
- output dims = max(1, round(source dims * scale))。
- density clamp 到 [0,1]。
- nearest 用于 debug；默认 trilinear。
- 不要把 scaled cells 直接 resample 到 full canvas；只生成 enlarged animal cells，然后仍由 resetImportedCells center-pad 到 simulation state。
```

### UI

在 Animal Catalog 或 Lenia Simulation 面板加入：

```text
Imported cells scale: 1.0 ... 8.0
Resample: Nearest / Trilinear
Auto-scale R with cells scale
Load animal original
Load animal scaled
Apply cells original
Apply cells scaled
Apply rule only
```

为了不破坏现有按钮，建议：

```text
- 保留 Load initial state + rule = original cells + original params。
- 新增 Load scaled state + scaled rule。
- 保留 Apply cells only = original cells。
- 新增 Apply scaled cells only。
- 保留 Apply rule only。
```

### R 缩放规则

当 `auto_scale_R` 开启且加载 scaled animal rule 时：

```cpp
params.radius = animal.params.radius * scale;
```

第一版保持：

```text
mu 不变
sigma 不变
b 不变
T 不变
kn/gn 不变
```

后续可以实验：

```text
T *= scale
或 velocity-preserving / frame-rate-preserving conventions
```

但不要在本计划中默认改变 T。

### 验收

```text
1. 仍能加载 original Lenia3D animal。
2. 能加载 scaled cells 到 128/160/192/256 canvas 中。
3. scale=1.0 时与 original path 视觉一致。
4. scale=2/4 时 body 明显变大，R 可随 scale 自动变大。
5. scaled cells + scaled R 能运行，不要求稳定复刻原 animal。
```

---

## 不做的事项

```text
- 不调查 reference animal 在大 canvas 中不稳定的全部原因。
- 不追求 Lenia3D bit-exact parity。
- 不做 parameter search。
- 不做 CUDA Graphs。
- 不做 cuFFT callbacks。
- 不做 multi-channel。
```

---

## 建议 commit

```powershell
git add .
git commit -m "sim: polish lenia animal scaling and fft path"
```
