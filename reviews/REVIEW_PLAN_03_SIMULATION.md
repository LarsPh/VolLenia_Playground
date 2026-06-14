# Review — PLAN 03 Single-channel 3D Lenia simulation

Date: 2026-06-14
Repo reviewed: `LarsPh/VolLenia_Playground`

## Summary

PLAN 03 is in good shape and is ready to move into biological preset/cell import work.

The implementation now has the important GPU pipeline pieces:

```text
Lenia state:      dense float32 DeviceVolume
Convolution:      cuFFT R2C -> pointwise multiply -> cuFFT C2R
Update:           CUDA kernel with growth + clamp
Renderer upload: generic DeviceVolumeView -> CUDA 3D array/texture
Display:          existing CUDA PBO -> OpenGL texture/fullscreen triangle
```

## What looks correct

- CMake links `CUDA::cufft` and includes the simulation `.cu` files.
- `DeviceVolumeView` gives the renderer a generic volume source, so renderer is no longer tied to `SyntheticVolume`.
- cuFFT layout matches the project’s x-fastest linear layout:

```text
index = (z * ny + y) * nx + x
R2C/C2R plan dimensions = nz, ny, nx
spectrum size = nz * ny * (nx / 2 + 1)
```

- Inverse FFT output is scaled by `1 / (nx * ny * nz)` before growth.
- Kernel construction is normalized to sum to 1.
- The kernel is built as a circular convolution kernel centered at the origin by using wrapped distances, which is equivalent to building a centered kernel and rolling it to origin.
- Runtime source switching between synthetic and Lenia is clean.
- UI already exposes play/pause, single step, reset/regenerate, resolution, parameter preset, seed preset, R/T/mu/sigma, and render controls.

## Non-blocking issues / future cleanup

### 1. `invalid_flag` copies back to CPU every simulation step

Current implementation checks NaN/Inf every step by copying `invalid_flag` to CPU. This is useful during bring-up, but it serializes the stream and can hide true performance.

Keep it for now. Later add a debug checkbox:

```text
[ ] Validate NaN/Inf every step
```

Default off for performance exploration.

### 2. Growth/kernel IDs from Lenia3D are not fully represented yet

Current `LeniaParams` has `R`, `T`, `mu`, `sigma`, `b`, but does not yet store `kn` and `gn` from Lenia3D.

This matters because Lenia3D animals use coupled parameters such as:

```text
R, T, b, m, s, kn, gn, cells
```

PLAN 04 should add explicit kernel-core and growth-function enums/IDs. In Lenia3D’s JS implementation, `kn` selects the kernel core and `gn` selects the growth function.

### 3. Imported Lenia3D names are currently parameter presets, not full organisms

The current UI decouples parameter presets from procedural seeds. This is useful for exploration, but reference Lenia “organisms” should be loaded as coupled bundles:

```text
animal preset = cells + params + kernel core + growth function + metadata
```

PLAN 04 should support both:

```text
Load animal:       cells + params together
Apply cells only:  keep current rule params
Apply params only: keep current state/cells
Manual edit:       fine-tune after loading
```

### 4. Higher resolutions can be exposed cautiously

The simulation clamps volume size to 256. Since 128^3 is currently fast, UI can add:

```text
64, 96, 128, 160, 192, 256
```

Treat 192/256 as experimental options; cuFFT plan creation and memory use become more noticeable.

## Recommended next milestone

Proceed to PLAN 04:

```text
Lenia3D reference biology catalog + imported cell seeds
```

Main goal: quickly see more meaningful organisms while preserving the ability to mix-and-match cells and rule parameters.
