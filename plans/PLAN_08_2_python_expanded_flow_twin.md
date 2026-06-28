# PLAN 08.2 — Python Expanded/Flow Twin, ModelState Bridge, and C++ Performance Fixes

## Context

PLAN 08.1 added the C++ Expanded/Flow runtime. The new C++ path is capable of multi-channel, multi-kernel expanded additive updates and Flow-Lenia-style transport. It also added composite rendering.

The next step is to make this runtime usable for differentiable experiments and model search. That requires a PyTorch twin that follows the same ModelSpec schema, plus a bridge for Python-generated multi-channel states to be visualized in the C++ renderer.

## Goals

1. Build a PyTorch twin of C++ ModelSpec expanded/flow semantics.
2. Produce 2D/radial visualizations of smooth Gaussian-mixture kernels.
3. Add a ModelState bridge for multi-channel state export/import.
4. Add targeted C++ performance fixes discovered during PLAN 08.1.
5. Add basic parity and performance tests.
6. Add a minimal ModelSpec-aware search smoke using the existing search style.

## Non-goals

- Do not add resources, obstacles, nutrient, poison, or environment dynamics yet.
- Do not add parameter localization.
- Do not add general sigma reintegration.
- Do not add neural update rules.
- Do not replace or remove legacy Lenia3D animal support.
- Do not introduce a new search algorithm in this milestone.

## Required C++ fixes

### 1. Reintegration `dd` fix

The current C++ sigma=0.5 transport path computes `dd` from `flow_max` and does not respect `spec.flow.reintegration_dd`. Fix this.

Rules:

- `transport_sigma` remains fixed at 0.5 for Stage 2.
- If `spec.flow.reintegration_dd > 0`, use it.
- Else compute an automatic `dd` from `dt`, `flow_max`, and `transport_sigma`.
- Avoid `ceil(exact_integer + 1e-6)` causing a 27-offset gather to become 125 offsets.
- Log or expose the actual `dd` in status/debug output.

For sigma=0.5, a good default is:

```text
max_displacement = dt * flow_max
auto_dd = max(1, ceil(max_displacement + transport_sigma - 1e-6))
```

When applying the source moved center, clamp displacement using:

```text
ma = max(dd - transport_sigma, 0)
mu = p + clamp(dt * F, -ma, ma)
```

### 2. Mass diagnostics performance

Do not copy every matter channel to CPU every rendered frame.

Implement one of:

- GPU block reduction + copy small block-sum buffer; or
- compute mass diagnostics only every N frames; or
- make mass diagnostics optional and off by default for interactive runs.

Acceptance: 3-channel 128³ should no longer synchronize on full-channel CPU copies every frame.

### 3. Timing hooks

Add optional timing for ExpandedFlowSimulation stages:

```text
R2C channel FFTs
per-kernel multiply/C2R/growth
matter sum
Sobel A_sum
Sobel U
flow compute
transport
renderer upload
renderer raymarch
```

This can be a debug JSON/log or minimal UI section. It does not need to be beautiful.

## Required Python twin

Add a new package; suggested:

```text
python/vollenia_model/
  __init__.py
  spec.py
  kernels.py
  growth.py
  expanded_flow.py
  flow_transport.py
  visualization.py
  export_state.py
  parity.py
```

### ModelSpec parser

Read the same JSON schema as C++:

```text
format_version
model_type
name
simulation.dims / simulation.resolution
state.channels
state.render_channel
update.mode
update.dt
update.clip
update.flow
kernels
```

### State layout

Use:

```text
A[C, Z, Y, X]
```

### Kernel families

Implement:

- `smooth_gaussian_mixture`
- `legacy_shell`

Smooth kernel must match C++:

```text
q = distance / R
envelope = sigmoid(envelope_sharpness * (1 - q))
raw = sum(amplitude_i * exp(-0.5 * ((q-center_i)/width_i)^2))
K = envelope * raw
K = K / K.sum()
```

### Growth families

Implement:

- `gaussian`
- `polynomial_lenia3d`

### Expanded additive

Implement the FFT-efficient path:

```text
A_hat[c] = rfftn(A[c])
P_hat[k] = A_hat[src[k]] * K_hat[k]
P[k] = irfftn(P_hat[k])
U[dst[k]] += weight[k] * growth_k(P[k])
A_next[c] = clamp(A[c] + dt * U[c], 0, 1)
```

Support kernel chunks.

### Flow mode

Implement:

```text
A_sum = sum matter channels
U = expanded affinity
grad_U = Sobel3D(U)
grad_A = Sobel3D(A_sum)
alpha = clamp((A_sum / theta_A)^alpha_power, 0, 1)
F = (1-alpha) * grad_U - alpha * grad_A
F = clamp(F, -flow_max, flow_max)
A_next = target-centric reintegration sigma=0.5
```

Use target-centric gather with `torch.roll` or equivalent. Match C++ as closely as possible.

### torch.compile

Add optional compile support for:

- single step; and/or
- rollout chunk.

Do not compile file IO, plotting, JSON export, or logging.

Add a benchmark script that compares eager and compiled modes and reports max absolute difference.

## Kernel visualizations

Add:

```text
python/scripts/plot_modelspec_kernels.py
```

Required outputs:

```text
2D z-mid slice PNG
radial profile PNG
basis component profile PNG
CSV radial profile
```

This should work for all JSON files under `configs/modelspec/`.

## ModelState bridge

Add a multi-channel state format:

```json
{
  "format_version": 1,
  "model_spec": "relative/path/to/model.json",
  "layout": "channel-major-x-fastest",
  "dims": [nx, ny, nz],
  "channels": 3,
  "state_file": "state.f32",
  "render": {
    "composite": true,
    "render_channel": 0
  }
}
```

Binary layout:

```text
A[c,z,y,x] float32, x fastest, channel-major
```

Python:

- export ModelState from rollout snapshots;
- optionally export a series of ModelState manifests.

C++:

- load ModelState and reset `ExpandedFlowSimulation` state;
- allow inspecting Python-generated multi-channel states in the renderer;
- keep legacy catalog behavior intact.

## Search smoke

Add a minimal ModelSpec-aware search smoke, not a new search method.

It can reuse existing ideas:

```text
profile: maintain / move_shape_target-like
optimize initial logits
optionally optimize growth mu/sigma or kernel weights later
hard or ST-hard train modes if implemented
hard eval
export top state as ModelState
```

This should not become a new monolithic script. Reuse/refactor modules where practical.

## Acceptance checklist

- Existing C++ app launches and legacy paths still work.
- `configs/modelspec/*.json` still load in C++.
- C++ Flow transport uses correct `dd` / spec override.
- C++ mass diagnostics no longer full-copy all matter channels every frame.
- PyTorch can load `flow_three_channel_complex.json`.
- PyTorch can run one expanded additive step and one Flow step on CUDA.
- PyTorch kernel plots are generated for at least one ModelSpec.
- Python can export a ModelState.
- C++ can load/render a Python-exported ModelState.
- Timing benchmark JSON exists for eager/compiled Python step or rollout chunk.

## Suggested manual commands

```powershell
uv run python python/scripts/plot_modelspec_kernels.py --spec configs/modelspec/flow_three_channel_complex.json --out outputs/modelspec_kernel_plots/flow_three_channel_complex
```

```powershell
uv run python python/scripts/torch_modelspec_rollout.py --spec configs/modelspec/flow_three_channel_complex.json --size 64 --steps 16 --out outputs/modelspec_twin/flow_three_channel_smoke
```

```powershell
uv run python python/scripts/benchmark_modelspec_torch_compile.py --spec configs/modelspec/flow_single_channel.json --size 64 --steps 16 --out outputs/modelspec_twin/bench_flow_single
```

Then open the exported ModelState in the C++ app and inspect composite rendering.
