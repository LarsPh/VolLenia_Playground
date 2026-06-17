# PLAN 06 — PyTorch differentiable backend + C++ renderer bridge MVP

## Goal

Add a research-oriented PyTorch backend for differentiable Lenia rollouts while keeping the existing C++/CUDA playground as the high-resolution visualization and inspection backend.

This is not a full search milestone yet. It is the foundation for later Sensorimotor-Lenia-style search:

```text
PyTorch differentiable simulator
  -> metrics / toy losses / gradient sanity
  -> export snapshots + params
  -> C++ loads exported snapshots through the existing animal/cell catalog path
  -> human visual inspection + coarse consistency checks
```

## Why this comes before the C++ headless worker

For a paper-oriented route, the immediate question is not “can we evaluate lots of candidates quickly?” but:

```text
Can a Lenia-like 3D volumetric substrate be optimized through differentiable rollouts?
Can its metrics/losses push an organism toward target movement, compactness, obstacle avoidance, and future resource seeking?
Can promising states be inspected in the current C++ volume renderer from day one?
```

The C++ worker remains useful later for high-resolution replay, long-rollout validation, and large-scale non-differentiable evaluation. But Plan 06 should establish the differentiable research loop first.

## High-level architecture

```text
VolLenia_Playground/
  python/
    vollenia_diff/
      __init__.py
      params.py
      kernels.py
      simulator.py
      metrics.py
      objectives.py
      export_cpp.py
      io.py

    scripts/
      torch_smoke_rollout.py
      torch_toy_optimize.py
      export_torch_state_to_cpp.py

    tests/
      test_fft_step_shapes.py
      test_grad_nonzero.py
      test_export_cpp_manifest.py

  docs/
    research_search_references.md
    pytorch_cpp_bridge.md

  configs/
    lenia.default.json                 # add configurable animal_catalog path if practical

  outputs/
    diff_bridge/                       # ignored runtime outputs
```

## Minimal dependencies

Do not vendor Python dependencies. Add a lightweight requirements file:

```text
python/requirements.txt
```

Suggested contents:

```text
torch
numpy
matplotlib
pytest
```

If CUDA PyTorch needs a specific install command, document it but do not hard-code it into CMake.

## State layout convention

The Python backend should use tensor shape:

```text
A: [Z, Y, X]
```

This matches the C++ x-fastest `.f32` flattening:

```text
index = (z * ny + y) * nx + x
```

When exporting a PyTorch tensor to `.f32`, flatten in `[z][y][x]` order with x fastest.

## PyTorch Lenia step

Implement a single-channel 2D/3D Lenia step.

### Parameters

Use a shared-ish schema compatible with existing C++ naming:

```python
LeniaParams(
    R: float,
    T: float,
    m: float,      # also accepted as mu
    s: float,      # also accepted as sigma
    b: list[float],
    kn: int,       # kernel core ID
    gn: int,       # growth function ID
)
```

### Kernel generation

Support 2D and 3D, but Plan 06 should prioritize 3D.

For 3D, generate a wrapped radial kernel on `[Z, Y, X]`:

```text
dx = min(x, nx - x)
dy = min(y, ny - y)
dz = min(z, nz - z)
q = sqrt(dx^2 + dy^2 + dz^2) / R
```

Then:

```text
if q < 1:
  shell_position = q * len(b)
  shell_index = floor(shell_position)
  local_r = fract(shell_position)
  K = b[shell_index] * kernel_core(local_r, kn)
else:
  K = 0
normalize K so sum(K) = 1
K_hat = torch.fft.rfftn(K)
```

### Important FFT normalization note

C++ cuFFT inverse transforms are unnormalized, so the C++ code multiplies by `1/N` after C2R. PyTorch `torch.fft.irfftn` with default normalization already returns the normalized inverse. Therefore the PyTorch simulator should **not** multiply by `1/N` again unless it explicitly changes `norm`.

### Growth functions

Match C++/Lenia3D IDs:

```text
gn = 1: polynomial growth
  G(U) = 2 * max(0, 1 - (U-m)^2 / (9s^2))^4 - 1

gn = 2: Gaussian growth
  G(U) = 2 * exp(-(U-m)^2 / (2s^2)) - 1

gn = 3: step growth
  G(U) = +1 if abs(U-m) <= s else -1
```

### Update

For C++ consistency:

```python
A_next = clamp(A + G(U) / T, 0, 1)
```

For toy differentiable experiments, add an optional `clip_mode`:

```text
hard: torch.clamp
none: no clamp, only for gradient sanity
soft: optional later; do not make it the default
```

Plan 06 can use `hard` for consistency and `none` or careful low-density init for gradient sanity.

## Metrics MVP

Implement metrics in `metrics.py`. Distinguish between differentiable metrics and diagnostic metrics.

### Differentiable metrics

Use density-weighted soft measurements:

```text
mass = sum(A)
center_of_mass = sum(A * coords) / (mass + eps)
second_moment = sum(A * ||coord - COM||^2) / (mass + eps)
covariance = sum(A * outer(coord-COM, coord-COM)) / (mass + eps)
anisotropy = eigmax(covariance) / (eigmin(covariance) + eps)
target_distance = ||COM - target||
obstacle_overlap = sum(A * obstacle) / (mass + eps)
border_mass = mass near boundary / (mass + eps)
```

### Diagnostic metrics

These may be non-differentiable:

```text
active_voxels = count(A > threshold)
bbox_min/max for A > threshold
max_density
mean_density
mass_cv over rollout samples
COM displacement
speed in body units
```

Plan 06 does not need full blob-glider scoring yet, but it should expose these primitives.

## Toy losses MVP

Implement in `objectives.py`:

```text
L_target = ||COM(A_T) - target||^2
L_mass = soft penalty for mass outside [mass_min, mass_max]
L_compact = soft penalty for second_moment above threshold
L_obstacle = density * obstacle overlap
L_border = border mass penalty
```

Plan 06 toy optimization can use:

```text
L = L_target + λ_mass L_mass + λ_compact L_compact + λ_border L_border
```

Obstacle loss can be present but not required in the first script.

## Scripts

### `torch_smoke_rollout.py`

Purpose:

```text
Run a deterministic 32^3 or 64^3 rollout and export snapshots.
```

Required flags:

```text
--dim 3
--size 32
--steps 32
--snapshot-interval 8
--preset diguttome_saliens
--out outputs/diff_bridge/smoke_rollout
```

Outputs:

```text
manifest.json
snapshots/step_0000.f32
snapshots/step_0008.f32
...
metrics.json
```

### `torch_toy_optimize.py`

Purpose:

```text
Do a tiny differentiable optimization through a short rollout.
```

Start simple. Optimize either:

```text
initial_state logits
```

or a small subset of parameters, for example:

```text
m, s, b logits
```

Acceptance:

```text
1. loss decreases over a few iterations
2. at least one learned tensor/parameter has nonzero gradient
3. it exports the final state to the bridge catalog
```

### `export_torch_state_to_cpp.py`

Purpose:

```text
Export a tensor or rollout snapshot as a LeniaAnimalCatalog-compatible catalog.
```

Output format:

```text
outputs/diff_bridge/catalog.json
outputs/diff_bridge/cells/<slug>.f32
```

The catalog should contain at least:

```json
{
  "format_version": 1,
  "layout": "x-fastest",
  "animals": [
    {
      "id": 0,
      "slug": "torch_step_0032",
      "name": "PyTorch step 0032",
      "dims": [32, 32, 32],
      "cells_file": "cells/torch_step_0032.f32",
      "params": {
        "R": 10.0,
        "T": 10.0,
        "b": [1.0, 0.75, 0.5833333, 0.9166667],
        "m": 0.12,
        "s": 0.01,
        "kn": 1,
        "gn": 1
      }
    }
  ]
}
```

## C++ renderer bridge

Day-one bridge requirement:

```text
PyTorch exported .f32 snapshot can be loaded and viewed in the C++ renderer.
```

Use the existing `LeniaAnimalCatalog` / `CellVolumeFile` path. Add the smallest possible C++ config extension:

```json
{
  "lenia": {
    "animal_catalog": "configs/lenia3d_reference/animals.json"
  }
}
```

Default behavior should remain unchanged. If the path points to `outputs/diff_bridge/catalog.json`, the C++ app should load the PyTorch export catalog.

Optional but useful:

```text
- Add a UI line showing current catalog path.
- Add a “Reload catalog” button.
```

Do not build a full file picker.

## Coarse consistency checks

Plan 06 should not try to prove exact C++/PyTorch equivalence yet, but should add sanity hooks:

```text
1. Same parameter schema names.
2. Same x-fastest f32 export.
3. Same kernel core and growth formulas.
4. Same default hard clamp update.
5. Export PyTorch step 0 and step N snapshots for visual comparison in C++.
```

Numerical C++ step dumping can be a later worker milestone.

## Acceptance criteria

1. `python -m pytest python/tests` passes.
2. `torch_smoke_rollout.py` produces `.f32` snapshots and a catalog.
3. C++ app can load the exported catalog path and display a PyTorch snapshot.
4. `torch_toy_optimize.py` demonstrates nonzero gradients and a decreasing toy loss.
5. Metrics JSON includes at least mass, COM, second moment, covariance eigenvalues or anisotropy, max density, and target distance.
6. No custom CUDA autograd or C++ torch extension is added.
7. Existing C++ renderer / Lenia UI still works with the default Lenia3D catalog.

## Suggested commit

```powershell
git add .
git commit -m "research: add pytorch differentiable lenia bridge"
```
