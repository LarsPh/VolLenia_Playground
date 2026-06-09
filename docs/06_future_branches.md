# 06 — 后期研究分支草图

这个文件不是第一阶段执行计划，只是保证早期架构不要堵死后续方向。

## Branch A — Kernel zoo / multi-channel Lenia

目标：从 single-channel single-kernel 进入可探索的 Lenia lab。

候选功能：

```text
Gaussian shell kernels
beta-like radial kernels
negative lobes / zero-mean kernels
multi-kernel basis
channel mixing matrix
body / membrane / energy / inhibitor / pigment channels
```

建议实现方式：

```text
先做多个 fixed kernels
再做 UI kernel editor
最后考虑 learnable kernel params
```

## Branch B — Spectral/logit field Lenia

核心形式：

```text
Z(x,t) = band-limited field
A(x,t) = sigmoid(Z(x,t))
Lenia growth computed on A
project growth/update back to spectral band
```

直觉目标：

```text
更平滑
频率预算约束复杂度
更像整体软体/弹性体
更适合 volume renderer 显示内部连续形变
```

建议先做 2D prototype，验证行为后再做 3D。

## Branch C — Environment / Eco Lenia

核心：

```text
A: organism density
N: nutrient field
P: poison / obstacle field
metabolic cost
resource consumption
bounded growth
```

目标不是让系统长成指定形状，而是在环境压力下持续存在。

建议先做 2D：

```text
更快
更容易可视化
更适合做 metrics/search
```

## Branch D — Learnable / Neural Lenia

候选层级：

```text
1. learnable radial kernel params
2. learnable growth curve
3. small MLP growth: Gθ(A, U, env)
4. flow velocity network: vθ(A, U, ∇U, env)
5. local rule-code/species embedding
```

训练/搜索建议：

```text
先用 random search / CMA-ES / MAP-Elites
不要一开始用复杂 RL
不要一开始在 3D 训练
```

可复用第一阶段的 metrics/logging/rendering。

## Branch E — High-end renderer backend

候选：

```text
OptiX backend for volume + mesh + shadows + scattering
Falcor backend for research render graph / RTX features
pbrt-v4 export path for offline cinematic renders
NanoVDB sparse volume renderer
```

这些都在第一阶段之后考虑。第一阶段的 CUDA renderer 不应依赖这些后端。
