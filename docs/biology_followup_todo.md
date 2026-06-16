# Biology follow-up TODO update — after PLAN 04

This file condenses the immediate follow-up work from `vollenia_lenia3d_followup_todo_v3.md` into an implementation-oriented checklist.

## Immediate PLAN 05 scope

### 1. Invalid flag validation simplification

- Remove `Validate NaN/Inf every step` from GUI.
- Remove config field if present.
- Always validate inside update kernel.
- Read `invalid_flag_` once per `simulateSteps(...)` call, not once per inner simulation step.
- If step-level localization is needed, use `steps_per_frame = 1`.

### 2. In-place spectrum multiply

Replace:

```text
state_spectrum_ + kernel_spectrum_ -> potential_spectrum_
```

with:

```text
state_spectrum_ *= kernel_spectrum_
```

Then run C2R from `state_spectrum_` into `potential_`.

Expected benefit:

```text
- one fewer packed complex buffer
- one fewer write target in spectrum multiply
- simpler memory footprint before higher resolutions
```

### 3. Keep spatial_kernel_

Do not remove or reuse `spatial_kernel_` for now. It keeps kernel rebuild/debug clean and is useful for future visualization.

### 4. Kernel visualization

Add Python static visualization:

```text
scripts/plot_lenia_kernel_profiles.py
```

Outputs:

```text
radial shell profile
FFT-origin 2D slice
fftshift-centered 2D slice
```

This is the first step toward understanding animal-specific `R + b + kn` fingerprints.

### 6. Animal + canvas linked scaling experiment

Add imported cells scaling:

```text
nearest / trilinear
scale factor
optional R *= scale
```

Keep `T`, `mu`, `sigma`, `b`, `kn`, and `gn` unchanged for the first pass.

Important design choice:

```text
scale the imported animal cell volume, then center-pad into the simulation canvas.
Do not stretch directly to the full canvas.
```

## Not in PLAN 05

### 5. Full instability investigation

Do not spend time trying to exactly reproduce Lenia3D behavior across canvas sizes. Many small numerical/backend differences can matter, and VolLenia does not need to be bit-exact.

### 7. Full animal stability evaluation

Keep this for parameter search. It will become part of a batch evaluator with mass, active voxels, bounding box, center of mass, speed, and stability scoring.

### 8. CUDA Graphs

Later. Useful once simulation step sequence stabilizes. Candidate captured sequence:

```text
R2C -> in-place multiply -> C2R -> update
```

Recapture when resolution/buffers/plans change.

### 9. cuFFT callbacks

Later than CUDA Graphs. First attempt should be a non-padding R2C store callback that multiplies by `kernel_spectrum_` during R2C store. Avoid padded in-place layout changes until profiling proves the complexity is justified.

## Relationship to future parameter search

PLAN 05 improves the manual exploration loop:

```text
bigger imported cells
scaled R
kernel profile images
lower memory footprint
cleaner validation behavior
```

The next research-oriented milestone can build batch search on top of this.
