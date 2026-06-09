# 02 — 路线图与里程碑

## Milestone 0 — Repo bootstrap

目标：创建可构建的空项目骨架。

输出：

```text
CMake project
src/main.cpp
basic GLFW/OpenGL window
Dear ImGui panel
CUDA runtime smoke test
.gitignore
README
configs skeleton
```

验收标准：

```text
cmake configure 成功
build 成功
运行后出现窗口和 ImGui 面板
面板显示 CUDA device name / app frame time
```

对应任务卡：

```text
plans/PLAN_00_bootstrap_repo.md
codex_prompts/CODEX_00_bootstrap_repo.md
```

---

## Milestone 1 — CUDA/OpenGL PBO smoke test

目标：验证 CUDA 能写 OpenGL PBO 并显示。

输出：

```text
OpenGL PBO
cudaGraphicsGLRegisterBuffer
CUDA kernel writes gradient/test pattern
fullscreen display
resize handling
```

验收标准：

```text
窗口显示 CUDA kernel 生成的动态图像
无 CPU readback
resize 不崩溃
Debug/Release 都能跑
```

对应任务卡：

```text
plans/PLAN_01_cuda_gl_pbo_smoke_test.md
codex_prompts/CODEX_01_cuda_gl_pbo_smoke_test.md
```

---

## Milestone 2 — Volume renderer MVP

目标：移植/重写 CUDA volume ray marcher，渲染 synthetic 3D volume。

输出：

```text
CUDA 3D texture object
synthetic sphere / noise / shell volume generator
ray-box intersection
fixed-step raymarch
emission-absorption alpha compositing
transfer function params
camera orbit
```

验收标准：

```text
能看到半透明 3D volume
ImGui 可调 step_size / density_scale / threshold / brightness
可以切换 synthetic volume presets
renderer 和 simulation 尚未耦合
```

对应任务卡：

```text
plans/PLAN_02_volume_renderer_mvp.md
codex_prompts/CODEX_02_volume_renderer_mvp.md
```

---

## Milestone 3 — Single-channel 3D Lenia simulation MVP

目标：接入 cuFFT，跑最基本的 3D Lenia。

输出：

```text
DeviceVolume
LeniaRule
KernelGenerator
CufftContext
R2C/C2R 3D convolution
growth update kernel
state rendered by volume renderer
```

验收标准：

```text
128^3 single-channel volume 能连续 step
renderer 显示当前 state
UI 可暂停/单步/重置
UI 可调 dt, μ, σ, kernel radius, shell weights preset
每步无 CPU readback
```

对应任务卡：

```text
plans/PLAN_03_lenia_sim_mvp.md
codex_prompts/CODEX_03_lenia_sim_mvp.md
```

---

## Milestone 4 — Lenia3D reference import

目标：参考 Lenia3D 的 kernel/growth/animals 数据，导入一批 seed 和参数。

输出：

```text
configs/lenia3d_reference/*.json
RLE 或 simplified seed importer
kernel core parity notes
growth function parity notes
animal/pattern selection UI
```

验收标准：

```text
至少导入 3 个 Lenia3D-style preset
能从 UI 选择 preset 并 reset
能保存当前参数为 JSON
```

对应任务卡：

```text
plans/PLAN_04_lenia3d_reference_import.md
```

---

## Milestone 5 — Renderer polish

目标：让 playground 有足够视觉反馈。

功能候选：

```text
gradient normals
rim lighting
MIP mode
first-hit debug mode
slice view
transfer function editor
screenshot/video frame dump
camera bookmarks
```

验收标准：

```text
同一 simulation state 可以用多种 render mode 检查
能产出好看的截图/短视频帧序列
```

---

## Milestone 6 — Kernel zoo + multi-channel skeleton

目标：从 single kernel 进入探索性 playground。

功能候选：

```text
Gaussian shell kernel
beta-like radial kernel
negative-lobe / zero-mean kernel
multi-kernel basis
2-4 channel state
channel mapping to color/alpha
```

验收标准：

```text
可以在 UI 中切换 kernel family
可以渲染不同 channel 或 channel mix
```

---

## Milestone 7 — Experiment harness

目标：让参数搜索和结果记录变得可重复。

功能候选：

```text
headless rollout mode
metrics: mass, center, compactness, spectral entropy, activity
JSON log
seeded random init
batch parameter sweep
PNG/frame dump
```

验收标准：

```text
一条命令可以跑 N 个参数组合并输出 metrics + preview frames
```

---

## Milestone 8A — Spectral/logit Lenia branch

建议先做 2D prototype，再搬回 3D。

核心：

```text
Z field -> A = sigmoid(Z)
Lenia growth on A
project growth back to spectral band
update Z
```

观察点：

```text
是否更平滑
是否更稳定
是否更像整体弹性体/软体
频率预算如何影响复杂度
```

---

## Milestone 8B — Environment / Eco branch

建议先做 2D prototype。

核心：

```text
A organism density
N nutrient field
P poison/obstacle field
metabolic cost
resource consumption
quality-diversity/random search
```

观察点：

```text
资源限制是否避免爆炸/全灭
是否出现追逐食物/局部摄食
是否能定义比较稳的 interestingness metrics
```

---

## Milestone 9 — 3D Eco / Field integration

当主线稳定后，把 2D 中最有希望的 idea 搬到 3D。

候选组合：

```text
3D volume renderer + multi-channel Lenia
3D spectral-logit state
3D resource field
Flow-like conservative transport
learnable kernel/growth search
```

这不是第一阶段目标。
