# Codex Prompt — PLAN 08.2 PyTorch Expanded/Flow Twin + ModelState Bridge

You are working in `D:\projects\VolLenia_Playground`.

Implement PLAN 08.2. Read these first:

```text
plans/PLAN_08_2_python_expanded_flow_twin.md
docs/plan08_stage2_design_notes.md
src/model/ModelSpec.h/.cpp
src/sim/ExpandedFlowSimulation.h/.cu
src/sim/FlowTransport.h/.cu
src/render/CudaVolumeRenderer.h/.cu
configs/modelspec/*.json
```

Also use the FlowLenia reference if available locally:

```text
D:\projects\FlowLenia
```

Reference GitHub:

```text
https://github.com/erwanplantec/FlowLenia
```

## High-level objective

Add a PyTorch twin for C++ ModelSpec expanded/flow models and bridge Python-generated multi-channel states back into the C++ renderer. Include targeted C++ performance fixes that are likely to help current 3-channel 128³ Flow runtime.

Do not redesign the whole project. Do not break legacy Lenia/animal catalog/synthetic volume paths.

## Required C++ tasks

### 1. Fix Flow reintegration `dd`

Current `launchFlowTransportSigmaHalf(...)` computes `dd` from `flow_max` and ignores `spec.flow.reintegration_dd`. Fix this.

- Pass either `reintegration_dd` or the full FlowParams into the transport launcher.
- Compute automatic `dd` from `dt * flow_max` and `transport_sigma=0.5` if no override is given.
- Avoid exact integer values being bumped to the next integer by epsilon.
- Make actual `dd` visible in debug/status output if possible.

### 2. Reduce mass diagnostic stalls

`ExpandedFlowSimulation::matterMass()` currently copies whole matter channels to CPU. Replace or gate it.

Acceptable solutions:

- CUDA reduction to a small host buffer; or
- update mass diagnostics every N frames/steps; or
- optional debug-only mass diagnostics off by default.

Do not copy every channel volume to CPU every rendered frame in normal mode.

### 3. Optional timing hooks

Add minimal CUDA event timing for expanded/flow model stages if practical. Do not over-engineer the UI.

## Required Python tasks

Create a new package such as:

```text
python/vollenia_model/
```

Implement:

```text
spec.py
kernels.py
growth.py
expanded_flow.py
flow_transport.py
visualization.py
export_state.py
```

### 1. ModelSpec parser

Mirror the C++ JSON schema. Keep names and defaults compatible.

### 2. Kernel bank

Implement `smooth_gaussian_mixture` and `legacy_shell`.

The smooth kernel formula must match C++:

```text
q = distance / R
envelope = sigmoid(envelope_sharpness * (1 - q))
raw = sum(amplitude_i * exp(-0.5 * ((q-center_i)/width_i)^2))
K = envelope * raw
K = K / K.sum()
```

Use wrapped radial distance.

### 3. Expanded additive step

Use tensor shape:

```text
A[C, Z, Y, X]
```

Use FFT over spatial axes only. Reuse channel FFTs and chunk kernels if needed.

### 4. Flow step

Implement:

- Sobel3D matching C++ weights and `/32` normalization.
- Flow field formula.
- Target-centric reintegration with `sigma=0.5`.
- Torus border first. Wall border if easy, but do not block on wall parity.

### 5. torch.compile

Add optional `compile_step` / `compile_rollout_chunk` support. Compile only pure tensor code.

### 6. Kernel visualization

Add script:

```text
python/scripts/plot_modelspec_kernels.py
```

It must generate PNG plots and CSV radial profiles.

### 7. ModelState bridge

Implement Python export of multi-channel states:

```text
state.f32
state.json
```

JSON schema:

```json
{
  "format_version": 1,
  "model_spec": "relative/path/to/model.json",
  "layout": "channel-major-x-fastest",
  "dims": [nx, ny, nz],
  "channels": 3,
  "state_file": "state.f32",
  "render": {"composite": true, "render_channel": 0}
}
```

Implement C++ load path so the GUI can render this exported state. A simple `Open model state...` button is ideal. A config field is acceptable only if time is short.

### 8. Minimal search smoke

Add a small script that uses the ModelSpec PyTorch twin in a non-toy way:

```text
python/scripts/search_modelspec_flow_smoke.py
```

It should reuse existing metric/loss ideas where practical, run a short BPTT over a ModelSpec model, and export a ModelState that C++ can render. Do not implement OpenES or MAP-Elites here.

## Tests / verification

Add pytest coverage for:

- ModelSpec parsing;
- kernel normalization;
- smooth kernel finite values;
- expanded additive step shape/finite values;
- flow step shape/nonnegative finite values;
- reintegration approximate mass conservation;
- ModelState export manifest and file size.

Add manual command docs in a milestone retrospective.

## Acceptance commands

At minimum these should work:

```powershell
uv run python -m pytest python/tests
```

```powershell
uv run python python/scripts/plot_modelspec_kernels.py --spec configs/modelspec/flow_three_channel_complex.json --out outputs/modelspec_kernel_plots/flow_three_channel_complex
```

```powershell
uv run python python/scripts/torch_modelspec_rollout.py --spec configs/modelspec/flow_three_channel_complex.json --size 64 --steps 8 --out outputs/modelspec_twin/flow_three_channel_smoke
```

Then build C++:

```powershell
cmake --build --preset release
```

Open the exported ModelState in the C++ app and inspect composite rendering.

## Important constraints

- Preserve legacy Lenia paths.
- Keep C++ high performance in mind.
- Do not add resources/environment/neural update/general sigma/parameter localization in this milestone.
- If a design choice affects the user workflow or adds a major dependency, explain the tradeoff in the retrospective.
