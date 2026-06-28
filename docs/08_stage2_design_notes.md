# Plan 08 Stage 2 design notes

## 1. Why this stage exists

Stage 08.1 added a C++-side `ModelSpec` path with:

- multi-channel state;
- multi-kernel FFT convolution;
- smooth Gaussian-mixture kernels;
- expanded additive mode;
- Flow mode;
- Sobel3D gradients;
- target-centric sigma=0.5 reintegration;
- composite multi-channel volume rendering.

The next milestone should make this path usable for research/search. That requires a PyTorch twin and a state bridge back into the C++ renderer.

The goal is not to improve the search algorithm itself yet. The goal is to make sure the new model is:

1. inspectable;
2. differentiable in Python;
3. close enough to C++ for short rollouts;
4. fast enough to iterate;
5. exportable into C++ for visual inspection.

## 2. C++ review items to fix first

### 2.1 Reintegration `dd` bug / ignored `reintegration_dd`

Current C++ has `FlowParams::reintegration_dd`, but the transport launcher computes `dd` internally from `flow_max`, not from the spec value. This means the spec field is currently not respected.

Current code pattern:

```cpp
const int dd = std::max(1, static_cast<int>(std::ceil(flow_max + 1.0e-6f)));
flowTransportSigmaHalfKernel<<<...>>>(..., dt, flow_max, dd, border);
```

This is problematic for two reasons:

1. `dd` should depend on maximum displacement `dt * flow_max`, not just `flow_max`.
2. `+1e-6` makes exact integer values such as `flow_max=1.0` become `ceil(1.000001)=2`, which changes the gather stencil from 27 offsets to 125 offsets in 3D.

For sigma=0.5, a conservative formula is:

```text
max_displacement = dt * flow_max
computed_dd = ceil(max_displacement + transport_sigma)
```

Because Stage 1 supports only `transport_sigma = 0.5`, this is:

```text
computed_dd = ceil(dt * flow_max + 0.5)
```

Then allow the JSON to override it:

```text
if spec.flow.reintegration_dd > 0:
    dd = spec.flow.reintegration_dd
else:
    dd = computed_dd
```

But if the override is too small to cover the displacement, clamp the displacement according to the chosen `dd`:

```text
ma = dd - sigma
mu = p + clamp(dt * F, -ma, ma)
```

This matches the FlowLenia reference implementation style: it uses `ma = dd - sigma` as the upper bound of displacement and only gathers offsets in `[-dd, dd]`.

### 2.2 Mass diagnostics should not copy all channels to CPU every frame

Current `ExpandedFlowSimulation::matterMass()` copies each matter channel to host and sums on CPU. That is acceptable for debugging but can synchronize the GPU and move many MB per frame.

Replace it with one of these:

- a small CUDA reduction into block sums, then copy the block sums;
- or compute mass only every N rendered frames / only when a debug checkbox is enabled.

Stage 2 acceptance should include at least one of:

```text
1. no full-volume CPU copy per rendered frame, or
2. UI/config option to reduce mass diagnostic frequency.
```

### 2.3 Add CUDA timing for model path

Add optional CUDA event timing around:

```text
channel R2C FFTs
kernel multiply + C2R + growth accumulate
matter sum
Sobel A_sum
Sobel U per channel
flow field compute
reintegration transport
renderer upload
composite raymarch
```

A tiny timing overlay or JSON/debug log is enough. Do not optimize blindly.

### 2.4 Composite render performance

Current composite render uploads each channel into a separate 3D texture and samples each enabled channel at every ray step. This is semantically fine, but it multiplies texture fetches.

Do not implement packed float4 composite texture unless profiling shows the renderer is a major bottleneck. Keep this as a follow-up candidate:

```text
Device channel-major state -> pack first 4 render channels into float4 3D cudaArray -> one tex3D<float4> per sample
```

## 3. Current multi-channel render semantics

The current composite renderer is reasonable for a first component overlay:

```text
density_sum = sum(channel_density * channel_intensity)
color = weighted average of channel palette colors
alpha = 1 - exp(-density_scale * max(density_sum - threshold, 0) * step)
```

This makes overlapping channels blend into intermediate colors, like the supplied screenshot.

However, it has limitations:

- overlapping channels can wash into pastel colors;
- channel identity is partially lost inside dense overlaps;
- different channel scale distributions can make one channel dominate;
- there is only one global threshold/density scale.

Future render modes worth adding after Stage 2:

```text
1. additive emission per channel;
2. dominant-channel hue + total-density alpha;
3. per-channel threshold/gamma/density scale;
4. packed float4 texture path;
5. channel-isosurface / selected-channel debug slices.
```

Stage 2 should not get distracted by these unless they are very cheap.

## 4. PyTorch ModelSpec twin

### 4.1 Package layout

Add a package, for example:

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

Do not overload the old single-channel `vollenia_diff.simulator` too much. Keep legacy Lenia and expanded/flow model code separated.

### 4.2 State layout

Use the same conceptual layout as C++:

```text
A[c, z, y, x]
```

Each channel is contiguous in C++ and should be exported/imported as channel-major `.f32`.

### 4.3 ModelSpec parsing

Parse the same JSON schema as C++:

```text
format_version
model_type
name
simulation.resolution / simulation.dims
state.channels
state.render_channel
update.mode
update.dt
update.clip
update.flow.*
kernels[*]
```

Python should not invent a different schema.

### 4.4 Kernel bank

Implement:

```text
legacy_shell
smooth_gaussian_mixture
```

The smooth kernel should match C++:

```text
q = distance / R
envelope = sigmoid(envelope_sharpness * (1 - q))
raw = sum(amplitude_i * exp(-0.5 * ((q - center_i) / width_i)^2))
K = envelope * raw
K /= K.sum()
```

Use wrapped distance / torus distance like C++ kernel bank.

### 4.5 FFT update

Efficient PyTorch structure:

```python
A_hat = torch.fft.rfftn(A, dim=(-3, -2, -1))        # [C, Z, Y, Xh]
for kernel_chunk:
    P_hat = A_hat[src_indices] * K_hat_chunk       # [Kchunk, Z, Y, Xh]
    P = torch.fft.irfftn(P_hat, s=shape, dim=(-3, -2, -1))
    B = growth(P) * weights
    U.index_add_(0, dst_indices, B)
```

Support `kernel_batch_size` so large K can be chunked.

### 4.6 Growth

Match C++:

```text
gaussian:
  2 * exp(-0.5 * ((u - mu) / sigma)^2) - 1

polynomial_lenia3d:
  2 * max(0, 1 - (u - mu)^2 / (9 sigma^2))^4 - 1
```

### 4.7 Sobel3D

Match the C++ Sobel kernel:

```text
derivative axis: -1, 0, 1
smoothing axes: 1, 2, 1
normalization: /32
```

A direct convolution or roll-based implementation is fine. It should be differentiable.

### 4.8 Flow mode

Compute:

```python
A_sum = sum(matter channels)
grad_A = sobel3d(A_sum)
for c in matter channels:
    grad_U = sobel3d(U[c])
    alpha = clamp((A_sum / theta_A) ** alpha_power, 0, 1)
    F[c] = (1 - alpha) * grad_U - alpha * grad_A
    F[c] = clamp(F[c], -flow_max, flow_max)
    next[c] = reintegration_sigma_half(A[c], F[c])
```

### 4.9 Reintegration sigma=0.5 target-centric gather

Implement target-centric gather matching C++:

```text
for oz, oy, ox in [-dd, dd]^3:
    source = roll(state, offset)
    source_flow = roll(flow, offset)
    mu = source_position + clamp(dt * source_flow, -ma, ma)
    weight = prod(max(1 - abs(target_position - mu), 0))
    next += source * weight
```

Use torus border first. Wall border can be implemented if not too hard, but torus parity is the priority.

## 5. ModelState bridge

Python search cannot be useful unless its multi-channel states can be inspected in the C++ renderer.

Add a minimal ModelState format:

```json
{
  "format_version": 1,
  "model_spec": "relative/or/absolute/path/to/model.json",
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

C++ additions:

```text
ModelStateLoader
ExpandedFlowSimulation::resetState(DeviceMultiVolumeView or DeviceMultiVolume&&)
GUI: Open model state... / Reload state
```

If full GUI support is too much, a config-driven state path is acceptable for Stage 2, but C++ must be able to render Python-exported multi-channel states.

## 6. Kernel visualization

Add:

```text
python/scripts/plot_modelspec_kernels.py
```

It should produce:

```text
outputs/modelspec_kernel_plots/<model_name>/kernel_<i>_slice.png
outputs/modelspec_kernel_plots/<model_name>/kernel_<i>_radial_profile.png
outputs/modelspec_kernel_plots/<model_name>/kernel_<i>_basis_components.png
outputs/modelspec_kernel_plots/<model_name>/summary.png
```

This is required. The new Gaussian-mixture kernels must be visually inspectable.

## 7. Search smoke

Do not implement a new search algorithm.

Add a minimal ModelSpec-aware search smoke that reuses the existing ideas:

```text
initial state logits / seed state
ModelSpec loaded from JSON
profile: maintain / move_shape_target-like
BPTT over expanded_additive or flow model
hard/ST train modes if useful
hard eval
export ModelState for C++ renderer
```

This does not need to be high-performing. It just proves the current Sensorimotor-style search path can target the new model.

## 8. Acceptance criteria

Minimum acceptance:

```text
1. Existing C++ legacy Lenia and ModelSpec presets still launch.
2. C++ reintegration dd uses dt/flow_max/sigma or spec override; no accidental 125-offset path for dt=0.14, flow_max=1 unless requested.
3. C++ mass diagnostics no longer copy all channels to CPU every rendered frame.
4. Python can load configs/modelspec/flow_three_channel_complex.json.
5. Python can build kernel bank and plot 2D/radial kernel visualizations.
6. Python can run one expanded additive step and one flow step on CUDA.
7. Python can export a multi-channel ModelState.
8. C++ can load/render that exported ModelState.
9. At least one short Python rollout exported to C++ visually matches expected channel structure.
10. A benchmark JSON reports eager/compiled timing for step or rollout_chunk.
```

Stretch:

```text
- Python vs C++ one-step max_abs_error / mass_error parity report.
- Packed composite texture prototype if profiling proves renderer-bound.
- ModelSpec-aware BPTT search smoke with exported top candidate.
```
