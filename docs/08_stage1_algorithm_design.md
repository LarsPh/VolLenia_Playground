# Plan 08 Stage 1 Algorithm Design — C++ Expanded/Flow Core

## Scope

This milestone is C++ first. Python parity and differentiable search are intentionally deferred to Stage 2.

Implement:

1. `ModelSpec` JSON schema for multi-channel, multi-kernel models.
2. C++ multi-channel `DeviceMultiVolume` storage.
3. Smooth Gaussian-mixture kernel generation.
4. Expanded additive update mode.
5. Flow mode with 3D Sobel gradients and target-centric reintegration tracking at `transport_sigma = 0.5`.
6. C++ renderer integration for selected render channel.
7. Basic ModelSpec presets and kernel visualization/debug output.

Do not implement yet:

- General `transport_sigma != 0.5` reintegration.
- Parameter localization.
- Neural update rules.
- Resource/nutrient/obstacle channels.
- Python parity/search migration.
- Source-centric atomic splat backend.

## Design principle

The new model must not be bolted onto legacy `LeniaSimulation` as a pile of conditionals. Keep legacy simulation for Lenia3D replay, and add a new model layer:

```text
src/model/ModelSpec.h/.cpp
src/model/KernelSpec.h
src/sim/DeviceMultiVolume.h/.cu
src/sim/ExpandedFlowSimulation.h/.cu
src/sim/KernelBank.h/.cu
src/sim/FlowTransport.h/.cu
```

The C++ renderer should continue consuming `DeviceVolumeView` for one selected channel, but simulation state may be multi-channel.

## ModelSpec v1

Draft schema:

```json
{
  "format_version": 1,
  "model_type": "expanded_flow",
  "state": {
    "channels": [
      { "name": "body", "role": "matter" },
      { "name": "hidden0", "role": "hidden_reserved" }
    ],
    "render_channel": 0
  },
  "update": {
    "mode": "expanded_additive",
    "dt": 0.1,
    "clip": "hard",
    "flow": {
      "theta_A": 1.0,
      "alpha_power": 2.0,
      "flow_max": 1.0,
      "transport_sigma": 0.5,
      "reintegration_dd": 1,
      "border": "torus",
      "gradient": "sobel3d"
    }
  },
  "kernels": [
    {
      "name": "body_self_0",
      "src": 0,
      "dst": 0,
      "weight": 1.0,
      "family": "smooth_gaussian_mixture",
      "R": 12.0,
      "basis": [
        { "center": 0.25, "width": 0.08, "amplitude": 1.0 },
        { "center": 0.55, "width": 0.12, "amplitude": 0.5 }
      ],
      "envelope_sharpness": 10.0,
      "growth": { "family": "gaussian", "mu": 0.12, "sigma": 0.015 }
    }
  ]
}
```

Also support `family = "legacy_shell"` for conversion/debug only, but the new default should be `smooth_gaussian_mixture`.

## Multi-channel / multi-kernel FFT path

State layout:

```text
A[c, z, y, x], x fastest
```

Efficient convolution:

1. Batched R2C FFT for each source channel:

```text
Ahat[C, z, y, xh]
```

2. For each kernel `k`:

```text
Phat[k] = Ahat[src[k]] * Khat[k]
```

3. Batched C2R inverse FFT for kernel potentials:

```text
P[k, z, y, x]
```

4. Growth and accumulate:

```text
U[dst[k]] += weight[k] * G_k(P[k])
```

Do not try to pre-sum kernels before inverse FFT if growth is nonlinear.

Memory policy:

- Stage 1 may allocate full `P[K]` for simple implementation.
- Add `kernel_batch_size` field to code structure, even if default equals `K`.
- Future 256^3 / large K can chunk kernels.

## Growth functions

Support:

- `polynomial_lenia3d`: legacy Lenia3D polynomial.
- `gaussian`: Flow/Sensorimotor-style default.

Keep `step` only if legacy replay needs it; do not expose it as a new search default. Remove or hide old unused “stepwise” experimental names if present.

## Smooth Gaussian-mixture kernel

For offset radius `r = ||x||`:

```text
q = r / R
envelope = sigmoid(envelope_sharpness * (1 - q))
raw_K = envelope * sum_i amplitude_i * exp(-0.5 * ((q - center_i) / width_i)^2)
K = raw_K / sum(raw_K)
```

Implementation notes:

- Build spatial kernel in host or CUDA; Stage 1 can build on host for simplicity if not a per-frame operation.
- Normalize in double on CPU if host-side; copy to device and FFT.
- Add debug output for radial profile and a 2D mid-slice image.

## Kernel visualization

Add a C++ or Python-free C++-generated dump path:

```text
outputs/kernel_debug/<model_name>_<kernel_name>_profile.csv
outputs/kernel_debug/<model_name>_<kernel_name>_slice_zmid.f32
```

At minimum, CSV columns:

```text
radius_bin, mean_value, max_value, sample_count
```

If it is easier to use existing renderer, provide a synthetic volume source from kernel slice or kernel volume.

## Flow mode

Expanded affinity first:

```text
U[c] = sum over kernels targeting c: h_k * G_k(K_k * A[src_k])
```

Then:

```text
A_sum = sum matter channels A[c]
alpha = clamp((A_sum / theta_A)^alpha_power, 0, 1)
F[c] = (1 - alpha) * grad(U[c]) - alpha * grad(A_sum)
```

Use 3D Sobel gradients in Stage 1, not central difference, unless profiling shows severe cost.

## 3D Sobel gradient

Implement a CUDA kernel that computes `grad_x, grad_y, grad_z` with periodic boundary.

Use separable weights conceptually:

```text
derivative: [-1, 0, 1]
smoothing:  [1, 2, 1]
```

For `grad_x`, derivative along x and smoothing along y,z. Normalize by a constant; exact scale can be absorbed by `dt/flow_max`, but must be consistent between U and A gradients.

Naive one-pass 27-neighbor implementation is acceptable for Stage 1 because FFT remains the main cost. Keep a TODO for separable optimization if profiling demands it.

## Reintegration tracking, sigma = 0.5

Stage 1 transport uses target-centric gather for consistency with FlowLenia reference and future general sigma.

For each target voxel `p`, gather contributions from source offsets in `[-dd,dd]^3`:

```text
source = p - offset
mu_source = source_center + clamp(dt * F[source], -ma, ma)
weight = overlap(target_cell(p), uniform_cube(mu_source, sigma=0.5))
A_next[p] += A[source] * weight
```

For `sigma = 0.5`, overlap reduces to tri-linear weights. With component-wise flow bound `flow_max`, choose:

```text
dd = ceil(flow_max + sigma - 0.5 + epsilon)
```

For sigma=0.5:

```text
dd = ceil(flow_max + epsilon)
```

Examples:

```text
flow_max = 0.5 -> dd = 1
flow_max = 1.0 -> dd = 1
flow_max = 1.5 -> dd = 2
```

Reference FlowLenia uses a target-centric `roll`/offset gather with `dd`, `sigma`, and a clipped displacement `ma = dd - sigma`.

Stage 1 must guard:

```text
transport_sigma must be 0.5
border must be torus or wall; torus default
```

## Source vs target backend

Target-centric gather:

- No atomics.
- Deterministic accumulation order.
- Directly generalizes to arbitrary sigma via overlap weights.
- More arithmetic: for `dd=1`, 27 offsets per output; for `dd=2`, 125 offsets.

Source-centric push:

- For sigma=0.5, only 8 target writes per source voxel.
- Needs atomicAdd in C++ CUDA.
- Nondeterministic floating accumulation.
- General sigma still needs many target writes and atomics.

Stage 1 uses target-centric gather. If profiling shows transport dominates, add source-centric tri-splat as an optional C++ backend later.

## Hidden channels

Stage 1 does not implement carried hidden channels, but ModelSpec roles must reserve room:

```text
matter          transported by flow
hidden_reserved not used yet
static_env      not transported
render_only     not used by update
```

Future rule:

- Hidden channels attached to matter/rule-code should be transported with matter using mass-weighted mixing.
- Static environment channels should not be transported; they update by diffusion/reaction or remain fixed.

## Acceptance tests

1. Existing legacy Lenia3D catalog still loads and renders.
2. A new ModelSpec preset loads in C++.
3. Expanded additive with one channel and one kernel produces visible evolution.
4. Expanded additive with two kernels produces finite values and no crashes.
5. Flow mode preserves total mass within tolerance over 10-50 steps.
6. Flow mode renders selected body channel.
7. Kernel debug CSV/slice is written and sane.
8. CMake Release build passes.
