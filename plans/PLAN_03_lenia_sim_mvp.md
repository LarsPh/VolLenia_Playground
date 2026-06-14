# PLAN 03 — Single-channel 3D Lenia simulation MVP

## 目标

接入 **CUDA + cuFFT**，完成 single-channel 3D Lenia simulation，并把当前状态 `A(x,y,z)` 交给现有 CUDA volume renderer 显示。

这一阶段的重点是：

```text
1. cuFFT 数据布局正确
2. kernel 归一化正确
3. inverse FFT 归一化正确
4. renderer 不绑定 LeniaRule，仍然只消费 generic float volume
5. 有足够好的 procedural test states，可以肉眼看到动态变化
```

不要在这一阶段导入 `animals3D.js` 的 RLE 动物库；那是 Plan 04。Plan 03 可以参考 Lenia3D 的参数和 kernel/growth 形式，但初始状态先用 procedural seeds。

---

## 本地参考

Lenia3D 本地参考：

```text
D:\projects\Lenia3D\src\core\LeniaEngine.js
D:\projects\Lenia3D\src\core\KernelGen.js
D:\projects\Lenia3D\src\core\FftUtils.js
D:\projects\Lenia3D\src\data\animals3D.js
```

核心等价逻辑：

```text
A -> FFT
Ahat * Khat -> Uhat
IFFT -> U
G(U; mu, sigma)
A = clamp(A + dt * G(U), 0, 1)
```

Lenia3D 中 `T` 对应 timestep 分母：

```text
dt = 1 / T
```

---

## 目标文件

建议创建/修改：

```text
src/core/VolumeDesc.h
src/core/CudaCheck.h
src/sim/CufftCheck.h
src/sim/DeviceVolume.h
src/sim/DeviceVolume.cu
src/sim/LeniaParams.h
src/sim/LeniaSimulation.h
src/sim/LeniaSimulation.cu
src/sim/LeniaSeeds.h
src/sim/LeniaSeeds.cu
src/render/CudaVolumeRenderer.h
src/render/CudaVolumeRenderer.cu
src/render/SyntheticVolume.h
src/render/SyntheticVolume.cu
src/app/App.h
src/app/App.cpp
src/app/UiPanel.h
src/app/UiPanel.cpp
configs/lenia.default.json
CMakeLists.txt
```

如果想减少重构，可以先保留现有文件名，但必须满足下文的数据流和接口要求。

---

## CMake 要求

新增 cuFFT 链接：

```cmake
target_link_libraries(VolLenia_Playground PRIVATE
    CUDA::cudart
    CUDA::cufft
    ...
)
```

新增 `.cu` 文件必须进入 `add_executable(...)`。

---

## 数据布局：非常重要

当前 renderer / synthetic volume 使用 linear layout：

```cpp
index = (z * ny + y) * nx + x;
```

也就是 **x 是最快变化维度**。Plan 03 必须沿用这个布局，避免 simulation 和 renderer 方向不一致。

cuFFT 3D plan 建议这样创建：

```cpp
cufftPlan3d(&r2c_plan, nz, ny, nx, CUFFT_R2C);
cufftPlan3d(&c2r_plan, nz, ny, nx, CUFFT_C2R);
```

频域 packed size：

```text
spectrum_count = nz * ny * (nx / 2 + 1)
```

不要写成 `nx * ny * (nz/2+1)`，除非你同时改变 linear layout。MVP 不建议改变当前 layout。

---

## Core data structures

建议新增：

```cpp
struct VolumeDesc {
    int nx = 128;
    int ny = 128;
    int nz = 128;
};

struct DeviceVolumeView {
    VolumeDesc desc;
    const float* data = nullptr;
};
```

把 `VolumeDesc` 从 `RenderParams.h` 移到 `src/core/VolumeDesc.h` 更干净。`RenderParams.h`、`SyntheticVolume.h`、simulation、renderer 都 include 这个 core header。

`DeviceVolume` RAII：

```cpp
class DeviceVolume {
public:
    void resize(VolumeDesc desc);
    void clear(float value = 0.0f);
    float* data();
    const float* data() const;
    DeviceVolumeView view() const;
    size_t voxelCount() const;
    size_t byteSize() const;
};
```

---

## Renderer refactor required

当前 `CudaVolumeRenderer::setVolume(const SyntheticVolume&)` 会把 renderer 绑到 synthetic volume。Plan 03 要改成 generic upload：

```cpp
void uploadVolume(DeviceVolumeView volume);
```

要求：

```text
1. 如果 desc 改变：重建 cudaArray + texture object。
2. 如果 desc 相同：复用 existing cudaArray + texture object，只做 cudaMemcpy3D DeviceToDevice。
3. renderer 不 include SyntheticVolume，不知道 LeniaSimulation。
4. SyntheticVolume 暴露 view()。
5. LeniaSimulation 暴露 currentStateView()。
```

这样 Plan 03 每帧可以直接：

```text
simulate N steps
renderer.uploadVolume(lenia.currentStateView())
renderer.render(mapped_pbo, camera, render_params)
```

不要每帧 destroy/recreate CUDA 3D texture。

---

## cuFFT buffers

MVP 使用 out-of-place R2C/C2R，避免 in-place padding 复杂度：

```text
A:     float         [nx * ny * nz]
U:     float         [nx * ny * nz]
Ahat:  cufftComplex  [nz * ny * (nx/2 + 1)]
Khat:  cufftComplex  [nz * ny * (nx/2 + 1)]
Uhat:  cufftComplex  [nz * ny * (nx/2 + 1)]
```

每个 simulation step：

```text
1. cufftExecR2C(A -> Ahat)
2. multiplySpectrum: Uhat[k] = Ahat[k] * Khat[k]
3. cufftExecC2R(Uhat -> U)
4. updateKernel:
     u = U[i] / (nx * ny * nz)
     g = 2 * exp(-((u - mu)^2) / (2 * sigma^2)) - 1
     A[i] = clamp(A[i] + (1/T) * g, 0, 1)
```

必须做 inverse FFT normalization：

```cpp
const float inv_n = 1.0f / float(nx * ny * nz);
```

否则 `U` 会按体素数放大。

---

## Kernel generator MVP

实现 Lenia3D-style radial shell kernel。参数：

```cpp
struct LeniaKernelParams {
    float radius = 10.0f;             // in voxels
    std::vector<float> shell_weights; // b
    KernelCore core = KernelCore::Polynomial;
};
```

MVP 可以在 CPU 上生成 spatial kernel，再上传 GPU 并用 cuFFT 生成 `Khat`。Kernel 只在参数变化时重建，CPU 生成成本可以接受。后续如果要做 learnable kernel，再迁移到 GPU generator。

归一化必须用 double 累加：

```text
for every voxel:
    compute kernel value
sum += value
K /= sum
```

如果 `sum <= 0`，报错，不要生成全零 kernel。

### 周期卷积对齐

两种实现方式任选一种，但要写清楚并保持一致：

**方式 A：centered then roll**

```text
先在中心位置 (nx/2, ny/2, nz/2) 生成 kernel
再 roll 到 origin
```

**方式 B：直接 origin-periodic distance**

```text
dx = min(x, nx - x)
dy = min(y, ny - y)
dz = min(z, nz - z)
distance = sqrt(dx*dx + dy*dy + dz*dz)
```

方式 B 等价于已经把 kernel 中心放到 FFT circular convolution 的 origin，代码更不容易出错。

### polynomial core

先实现：

```cpp
core(r) = pow(4*r*(1-r), 4), r in [0,1]
```

shell mapping：

```text
q = distance / radius
if q >= 1: K = 0
else:
  s = q * B
  shell_index = clamp(floor(s), 0, B-1)
  local_r = fract(s)
  K = shell_weights[shell_index] * core(local_r)
```

---

## Growth MVP

先实现 Gaussian growth：

```cpp
G(U) = 2 * exp(-((U - mu) * (U - mu)) / (2 * sigma * sigma)) - 1;
A = clamp(A + dt * G(U), 0, 1);
dt = 1 / T;
```

数值保护：

```text
sigma >= 1e-5
T >= 1
if result is NaN/Inf: set to 0 or previous value, and expose error/status
```

---

## Parameter presets

加入少量参数 preset，先不导入 RLE cells。可以参考 Lenia3D `animals3D.js` 里的开头几个 3D 动物参数：

```json
{
  "Diguttome saliens": {"R": 10, "T": 10, "b": [1.0, 0.75, 0.5833333, 0.9166667], "mu": 0.12, "sigma": 0.01},
  "Diguttome tardus":  {"R": 10, "T": 10, "b": [0.6666667, 1.0, 0.8333333],       "mu": 0.15, "sigma": 0.016},
  "Triguttome labens": {"R": 10, "T": 10, "b": [1.0, 0.4166667, 0.0833333, 0.1666667], "mu": 0.16, "sigma": 0.015}
}
```

这些参数配 procedural seeds 不保证复现原动物，只作为初期动态测试。

---

## Initial state presets

至少实现这些 seed：

```text
Centered random ball:
  半径内随机 smooth density，外部 0。

Asymmetric Gaussian cluster:
  多个偏心 Gaussian blobs，用于破坏球对称。

Shell + internal blobs:
  类似当前 LeniaPhantom，但作为 simulation 初始状态。

Impulse / small blob:
  用于 convolution sanity check。
```

随机数用 deterministic hash，不需要 cuRAND。

---

## UI / App integration

保留 synthetic renderer，用 source toggle 切换：

```text
Source: Synthetic / Lenia
```

Lenia UI：

```text
Play/Pause
Single step
Reset seed
Regenerate seed
Steps per frame: 0..32
Seed preset
Parameter preset
R
T
mu
sigma
shell weights preset display/edit later
Rebuild kernel button
Generation counter
Current volume size
```

当 source 为 Lenia：

```text
if playing: simulate steps_per_frame
if single_step: simulate 1 step
upload Lenia A to volume renderer
render
```

当 source 为 Synthetic：保持当前 Plan 02 行为。

---

## Debug / validation

MVP 至少显示：

```text
kernel status: built / dirty / error
simulation status: initialized / running / paused / error
generation count
last error
```

可选但推荐：少量 scalar stats，不要 full-volume CPU readback：

```text
mass estimate / min / max via GPU reduction
或者 Debug build 下偶尔抽样少量体素
```

必须保证：

```text
1. kernel spatial sum ≈ 1 before FFT
2. U after normalized IFFT 大致在 [0,1] 范围内
3. A after update 始终在 [0,1]
4. no NaN/Inf
```

---

## 约束

```text
1. 不导入 TensorFlow.js / WebGL code。
2. 不做 multi-channel。
3. 不做 animals3D RLE importer；Plan 04 再做。
4. 不引入 OptiX/Falcor/pbrt/Vulkan/DirectX。
5. 不 CPU readback full simulation volume。
6. 不把 simulation 主状态存成 cudaArray；主状态必须是 cuFFT-friendly linear buffer。
7. 不做 half precision / batched FFT / sparse volume。
```

---

## 验收标准

```text
1. 64^3、96^3、128^3 single-channel Lenia 都可以初始化并 step。
2. 128^3 可以连续运行，renderer 显示当前 A。
3. Synthetic / Lenia source toggle 可用。
4. Play/Pause/Step/Reset 可用。
5. 改变 R / b / kernel preset 后 kernel 可以重建。
6. 改变 mu / sigma / T 后不需要重建 FFT plan，只影响 update。
7. inverse FFT 正确归一化，U 不出现按 N 放大的错误。
8. A 始终 clamp 到 [0,1]，无 NaN/Inf。
9. renderer 通过 generic DeviceVolumeView 上传，不依赖 LeniaSimulation。
10. 没有 full-volume CPU readback。
```

---

## 建议 commit

```powershell
git add .
git commit -m "sim: add single channel 3d lenia mvp"
```
