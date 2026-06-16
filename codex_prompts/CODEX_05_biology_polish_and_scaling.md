# Codex prompt — 05 Biology polish, in-place FFT, kernel visualization, and scaled cells

You are working in `D:\projects\VolLenia_Playground`.

The repo should already have:

- CUDA/OpenGL PBO display
- CUDA volume renderer
- single-channel cuFFT Lenia simulation
- Lenia3D animal catalog import
- original imported animal cells + params loading

## Goal

Implement a short follow-up feature pass after PLAN 04:

1. Remove the NaN/Inf validation checkbox and switch to one invalid-flag CPU readback per `simulateSteps(...)` call.
2. Optimize cuFFT convolution memory by changing the spectrum multiply to in-place and deleting `potential_spectrum_`.
3. Keep `spatial_kernel_`; do not over-optimize kernel rebuild memory.
4. Add a Python kernel visualization script.
5. Add scaled imported animal cells with nearest/trilinear resampling and optional R scaling.

Do not implement parameter search in this step.

---

## Constraints

- Do not change the primary `DeviceVolume` layout: keep tightly packed x-fastest float volumes.
- Do not introduce cuFFT callbacks, CUDA Graphs, OptiX, Falcor, pbrt, Vulkan, DirectX, or multi-channel simulation.
- Do not delete `spatial_kernel_`.
- Do not chase Lenia3D bit-exact parity.
- Do not add Python dependencies to CMake.
- Do not require CPU readback of full volumes.

---

## Task 1 — validation simplification

Current UI likely has `Validate NaN/Inf every step`. Remove it.

Implementation requirements:

- Remove `LeniaConfig::validate_nan_inf_every_step`.
- Remove JSON loading for `validate_nan_inf_every_step`.
- Remove ImGui checkbox.
- Change `LeniaSimulation::simulateSteps(int steps, bool validate_nan_inf)` to `simulateSteps(int steps)`.
- In `simulateSteps`:
  - `cudaMemset(invalid_flag_, 0, sizeof(int))` once before the loop over `steps`.
  - Each update kernel always checks `!isfinite(next)` and writes to `invalid_flag_`.
  - After all steps, read `invalid_flag_` once with `cudaMemcpy`.
- Keep existing pause/report behavior when invalid values are seen.

If precise step localization is needed later, users can set `steps_per_frame = 1`.

---

## Task 2 — in-place spectrum multiply

Current path:

```text
R2C(state -> state_spectrum_)
multiply(state_spectrum_, kernel_spectrum_ -> potential_spectrum_)
C2R(potential_spectrum_ -> potential_)
```

Change to:

```text
R2C(state -> state_spectrum_)
multiplyInPlace(state_spectrum_ *= kernel_spectrum_)
C2R(state_spectrum_ -> potential_)
```

Implementation requirements:

- Delete `cufftComplex* potential_spectrum_` from `LeniaSimulation`.
- Delete its `cudaMalloc` and `cudaFree`.
- Replace the multiply kernel with:

```cpp
__global__ void multiplySpectrumInPlaceKernel(
    cufftComplex* spectrum,
    const cufftComplex* kernel,
    std::size_t count);
```

- In `simulateSteps`, call C2R with `state_spectrum_` as input.
- Keep `potential_` as a float volume.
- Run the app and confirm visual behavior still looks sane.

---

## Task 3 — keep spatial_kernel_

Do not remove `spatial_kernel_`. It is still useful for kernel rebuild clarity and future kernel visualization/debugging.

---

## Task 4 — Python kernel visualization script

Add:

```text
scripts/plot_lenia_kernel_profiles.py
```

Features:

- Read `configs/lenia3d_reference/animals.json`.
- Select animal by `--animal-index` or generate `--all` / `--limit N`.
- Support `--size 128` and `--output-dir outputs/kernel_profiles`.
- Generate:

```text
<slug>_radial_profile.png
<slug>_kernel_origin_slice.png
<slug>_kernel_centered_slice.png
```

- Use numpy/matplotlib only.
- Print a clear message if numpy/matplotlib is missing.
- Implement the same kernel core semantics as current C++:
  - kn=1 polynomial bump
  - kn=2 exponential bump
  - kn=3 step
  - kn=4 staircase
- Normalize kernel sum to 1.
- `origin_slice` is the FFT-origin wrapped view at z=0.
- `centered_slice` uses `fftshift` for human-readable centered view.

Do not wire this into the C++ app yet.

---

## Task 5 — scaled imported animal cells

Add a small scaling/resampling path for imported Lenia3D cells.

Recommended structure:

```text
src/io/CellResampler.h
src/io/CellResampler.cu
```

or an equivalent clean location.

Add an enum:

```cpp
enum class CellResampleMode {
    Nearest = 0,
    Trilinear,
};
```

Behavior:

- Input: `DeviceVolumeView source_cells`.
- Output: `DeviceVolume scaled_cells`.
- Output dims = `round(source_dims * scale)`, clamped to at least 1.
- `scale=1.0` should preserve original behavior as closely as possible.
- `Nearest` and `Trilinear` modes are required; default should be trilinear.
- Clamp density to `[0,1]`.
- Resample only the animal cell volume, not the full simulation canvas. The existing centered copy into the simulation state should still happen afterward.

UI additions:

- Imported cells scale slider, e.g. `1.0 ... 8.0`.
- Resample mode combo: Nearest / Trilinear.
- Checkbox: Auto-scale R with imported cells.
- Keep existing original buttons.
- Add scaled buttons:
  - `Load scaled state + scaled rule`
  - `Apply scaled cells only`

When loading scaled rule with auto-scale R:

```cpp
params.radius = animal.params.radius * scale;
```

Keep `T`, `mu`, `sigma`, `b`, `kn`, and `gn` unchanged in this plan.

Acceptance criteria:

- Original animal loading still works.
- Scaled animal loading works at 128/160/192/256 grids.
- `scale=1.0` matches original path approximately.
- `scale=2` or `scale=4` gives visibly larger organisms.
- Auto-scaled R updates the parameter UI.
- No full-volume CPU readback.

---

## Build/run

After implementing:

```powershell
cmake --build --preset release
.\build\Release\VolLenia_Playground.exe
```

Also test the visualization script:

```powershell
python scripts\plot_lenia_kernel_profiles.py `
  --manifest configs\lenia3d_reference\animals.json `
  --animal-index 0 `
  --size 128 `
  --output-dir outputs\kernel_profiles
```

Summarize:

- files changed
- memory buffer change from removing `potential_spectrum_`
- validation behavior change
- how to use scaled animal cells
- how to run the kernel visualization script
