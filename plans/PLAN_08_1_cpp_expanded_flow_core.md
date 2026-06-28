# PLAN 08 Stage 1 — C++ Expanded / Flow VolLenia Core

## Goal

Upgrade the C++ runtime from legacy single-channel Lenia toward a model core that can support Expanded Lenia and Flow-Lenia. This stage is C++ only. Python parity/search migration comes later.

## Non-goals

- No Python model rewrite in this stage.
- No OpenES / MAP-Elites integration.
- No resources, obstacles, food, poison.
- No parameter localization.
- No neural update rule.
- No general `transport_sigma != 0.5` implementation.

## Work items

### 1. ModelSpec JSON

Add:

```text
src/model/ModelSpec.h
src/model/ModelSpec.cpp
src/model/KernelSpec.h
```

Parse:

```text
channels
update mode: expanded_additive | flow
flow params
kernel specs
render channel
```

Keep schema backward-compatible with existing animal catalog; ModelSpec is a new path, not a replacement for old catalog yet.

### 2. Multi-channel device storage

Add:

```text
src/sim/DeviceMultiVolume.h
src/sim/DeviceMultiVolume.cu
```

Requirements:

```text
channel-major layout: A[c][z][y][x]
views for selected channel as DeviceVolumeView
clear/fill/copy channel
resize/destroy RAII
```

### 3. KernelBank

Add:

```text
src/sim/KernelBank.h
src/sim/KernelBank.cu
```

Support:

```text
legacy_shell
smooth_gaussian_mixture
```

For smooth Gaussian mixture, build normalized spatial kernels and FFT them with cuFFT.

Add debug dumps:

```text
outputs/kernel_debug/*.csv
outputs/kernel_debug/*.f32
```

### 4. Expanded additive simulation

Add:

```text
src/sim/ExpandedFlowSimulation.h
src/sim/ExpandedFlowSimulation.cu
```

Implement expanded additive mode:

```text
Ahat[c] = FFT(A[c])
P[k] = IFFT(Ahat[src[k]] * Khat[k])
U[dst[k]] += h[k] * growth_k(P[k])
A_next[c] = clamp(A[c] + dt * U[c], 0, 1)
```

Use batched cuFFT where practical. It is acceptable to implement kernel-batch loops first, but the code structure must allow chunking.

### 5. Flow mode

Implement:

```text
U from expanded affinity
grad_U with 3D Sobel
grad_A_sum with 3D Sobel
alpha = clamp((A_sum/theta_A)^alpha_power, 0, 1)
F = (1-alpha)*grad_U - alpha*grad_A_sum
A_next = reintegration_tracking(A, F)
```

Stage 1 transport:

```text
target-centric gather
transport_sigma = 0.5 only
periodic/torus default
```

### 6. C++ UI integration

Add a new source/mode, without breaking old source modes:

```text
Source: Synthetic | Lenia | ModelSpec
```

Minimum UI:

```text
Open model spec...
Reload current model spec
Update mode display
Channel count
Kernel count
Render channel combo
Play / pause / single step / reset
```

The existing volume renderer should render selected channel through `DeviceVolumeView`.

### 7. Presets

Add small presets:

```text
configs/modelspec/expanded_single_kernel.json
configs/modelspec/expanded_two_kernel.json
configs/modelspec/flow_single_channel.json
configs/modelspec/flow_two_channel.json
```

They do not need to be biologically impressive; they need to validate the model path.

### 8. Cleanup

Do not keep obsolete experimental growth/kernel names exposed in new UI. Legacy code can remain if needed, but new ModelSpec should expose only:

```text
growth: polynomial_lenia3d | gaussian
kernel: legacy_shell | smooth_gaussian_mixture
```

## Acceptance

- Existing app still builds and legacy Lenia mode still works.
- New ModelSpec presets load and render.
- Expanded additive mode runs finite for at least 50 steps.
- Flow mode approximately preserves body mass over 50 steps.
- Kernel debug dumps are produced.
- No Python search code is required to pass this milestone.
