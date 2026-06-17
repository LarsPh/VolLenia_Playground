# Codex prompt — 06 PyTorch differentiable Lenia backend + C++ bridge

You are working in `D:\projects\VolLenia_Playground`.

The repository already has a C++/CUDA/cuFFT 3D Lenia simulator, a CUDA volume renderer, a Lenia3D animal catalog importer, and imported cell scaling.

## Goal

Add a research-oriented PyTorch backend for differentiable Lenia experiments and an immediate bridge from PyTorch snapshots to the existing C++ volume renderer.

This milestone is not a full parameter-search system. It is the foundation for later Sensorimotor-Lenia-style search.

## Core requirements

1. Add a Python package under:

```text
python/vollenia_diff/
```

2. Implement a PyTorch 2D/3D Lenia simulator with:
   - FFT convolution via `torch.fft.rfftn` / `torch.fft.irfftn`
   - radial shell kernel generation
   - kernel cores matching C++ IDs where practical
   - growth functions matching C++/Lenia3D IDs
   - hard clamp update by default
   - differentiable metrics

3. Add scripts:
   - `python/scripts/torch_smoke_rollout.py`
   - `python/scripts/torch_toy_optimize.py`
   - `python/scripts/export_torch_state_to_cpp.py`

4. Add C++ bridge support:
   - read optional `animal_catalog` path from `configs/lenia.default.json`
   - default remains `configs/lenia3d_reference/animals.json`
   - allow PyTorch export catalog at `outputs/diff_bridge/catalog.json` to be loaded by the C++ app
   - optional: UI text showing catalog path and a “Reload catalog” button

5. Do not add custom CUDA autograd, LibTorch, pybind11, or a C++ torch extension.

## Python state layout

Use:

```text
A.shape == [Z, Y, X]
```

Export `.f32` with x fastest:

```python
A.contiguous().cpu().numpy().astype("<f4").tofile(path)
```

The catalog `dims` should be `[nx, ny, nz]` to match current C++ `VolumeDesc`.

## FFT normalization

Important: C++ cuFFT inverse is unnormalized and the C++ simulator multiplies by `1/N` after C2R. PyTorch `torch.fft.irfftn` default already gives the normalized inverse. Do not multiply by `1/N` a second time in PyTorch unless you explicitly use a different norm and document it.

## Implemented files

Suggested Python files:

```text
python/requirements.txt
python/vollenia_diff/__init__.py
python/vollenia_diff/params.py
python/vollenia_diff/kernels.py
python/vollenia_diff/simulator.py
python/vollenia_diff/metrics.py
python/vollenia_diff/objectives.py
python/vollenia_diff/export_cpp.py
python/vollenia_diff/io.py

python/scripts/torch_smoke_rollout.py
python/scripts/torch_toy_optimize.py
python/scripts/export_torch_state_to_cpp.py

python/tests/test_fft_step_shapes.py
python/tests/test_grad_nonzero.py
python/tests/test_export_cpp_manifest.py
```

Suggested C++ files to touch:

```text
src/app/App.h
src/app/App.cpp
src/app/UiPanel.cpp
configs/lenia.default.json
```

Only make minimal C++ changes needed for catalog path configurability.

## Metrics to implement

In `metrics.py`, implement at least:

```text
mass(A)
center_of_mass(A, coords)
second_moment(A, coords, com)
covariance(A, coords, com)
anisotropy(A)
max_density(A)
mean_density(A)
target_distance(A, target)
obstacle_overlap(A, obstacle)
border_mass(A)
rollout_summary(states)
```

For differentiable metrics use soft density-weighted formulas. Thresholded bounding boxes can be diagnostic-only and optional.

## Toy optimization

`torch_toy_optimize.py` should:

1. initialize a small 3D state, default 32^3
2. generate a kernel
3. roll out for a short horizon, e.g. 16-32 steps
4. optimize a simple differentiable objective, such as final COM toward a target plus mass/compactness penalties
5. print loss and gradient norm
6. export final state as a C++ animal catalog

Keep the toy task deliberately small and robust. It does not need to discover a real glider yet.

## Export bridge

The PyTorch export must create:

```text
outputs/diff_bridge/catalog.json
outputs/diff_bridge/cells/<slug>.f32
outputs/diff_bridge/metrics.json
```

The catalog must be compatible with the existing `LeniaAnimalCatalog`.

Then the user should be able to set:

```json
{
  "lenia": {
    "animal_catalog": "outputs/diff_bridge/catalog.json"
  }
}
```

or equivalent config to inspect the PyTorch state in the C++ renderer.

## Tests

Add pytest tests that do not require a GPU unless CUDA is available.

Required tests:

1. kernel shape and normalization for 2D/3D
2. one rollout step returns correct shape and finite values
3. at least one parameter or initial-state tensor receives a nonzero gradient in the toy loss
4. export manifest references an existing `.f32` file with expected byte size

## Acceptance criteria

- `python -m pytest python/tests` passes.
- `python/scripts/torch_smoke_rollout.py --size 32 --steps 16 --out outputs/diff_bridge/smoke` runs.
- `python/scripts/torch_toy_optimize.py --size 32 --iters 10 --out outputs/diff_bridge/toy` shows decreasing or at least non-flat loss and nonzero gradient norm.
- C++ app can load a PyTorch-exported catalog through the configured catalog path.
- Existing C++ default behavior with the Lenia3D catalog still works.

After implementation, summarize:
- files changed
- how to run tests
- how to run smoke rollout
- how to load PyTorch snapshot in C++ renderer
- any caveats about PyTorch/C++ numerical differences
