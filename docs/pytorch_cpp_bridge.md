# PyTorch ↔ C++ renderer bridge

## Purpose

From the first differentiable backend milestone, every PyTorch rollout should be inspectable in the existing C++ volume renderer.

This supports:

```text
1. human visual review
2. quick sanity checking
3. layout/axis/order validation
4. PyTorch/C++ rule comparison by eye
5. high-quality screenshots/videos from PyTorch-generated states
```

## Format

PyTorch exports a LeniaAnimalCatalog-compatible manifest plus `.f32` cells.

```text
outputs/diff_bridge/catalog.json
outputs/diff_bridge/cells/<slug>.f32
```

The `.f32` file is little-endian float32, x-fastest layout:

```text
index = (z * ny + y) * nx + x
```

The tensor layout in Python is:

```text
A[z, y, x]
```

## Example catalog

```json
{
  "format_version": 1,
  "layout": "x-fastest",
  "source": "pytorch_diff_backend",
  "animals": [
    {
      "id": 0,
      "source_index": 0,
      "slug": "toy_final",
      "code": "torch-toy",
      "name": "PyTorch toy final",
      "dims": [32, 32, 32],
      "cells_file": "cells/toy_final.f32",
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

## C++ config

Plan 06 should add an optional catalog path to `configs/lenia.default.json`:

```json
{
  "lenia": {
    "animal_catalog": "configs/lenia3d_reference/animals.json"
  }
}
```

To inspect PyTorch exports:

```json
{
  "lenia": {
    "animal_catalog": "outputs/diff_bridge/catalog.json"
  }
}
```

## Consistency notes

Do not expect bit-exact equality yet.

Expected differences:

```text
PyTorch irfftn normalization differs from raw cuFFT convention but should be accounted for in the implementation.
Floating point ordering differs.
Hard clamp may affect gradients.
C++ visualization may use larger render density/threshold settings.
```

Plan 06 only needs coarse visual bridge. Numerical step-dump comparison can come later with a headless worker.
