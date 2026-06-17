from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metrics import state_summary
from .params import LeniaParams


def apply_export_activation(state: torch.Tensor, mode: str) -> torch.Tensor:
    """Map an optimization state to a renderer-facing density field."""

    if mode == "raw":
        return state
    if mode == "clamp":
        return torch.clamp(state, 0.0, 1.0)
    if mode == "sigmoid":
        return torch.sigmoid(state)
    raise ValueError(f"Unknown export activation: {mode}")


def tensor_to_f32_file(state: torch.Tensor, path: Path) -> None:
    """Write [Z, Y, X] tensor as little-endian float32 with x-fastest layout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    array = state.detach().contiguous().cpu().numpy().astype("<f4", copy=False)
    array.tofile(path)


def dims_nx_ny_nz(state: torch.Tensor) -> list[int]:
    if state.ndim == 2:
        ny, nx = state.shape
        return [int(nx), int(ny), 1]
    if state.ndim == 3:
        nz, ny, nx = state.shape
        return [int(nx), int(ny), int(nz)]
    raise ValueError(f"Expected [Y, X] or [Z, Y, X] tensor, got shape {tuple(state.shape)}")


def write_catalog(
    states: dict[str, torch.Tensor],
    out_dir: Path,
    params: LeniaParams,
    *,
    source: str = "pytorch_diff_backend",
    metrics: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    animals: list[dict[str, Any]] = []
    for animal_id, (slug, state) in enumerate(states.items()):
        safe_slug = slug.replace("\\", "_").replace("/", "_")
        cells_path = cells_dir / f"{safe_slug}.f32"
        tensor_to_f32_file(state, cells_path)
        animals.append(
            {
                "id": animal_id,
                "source_index": animal_id,
                "slug": safe_slug,
                "code": f"torch-{animal_id}",
                "name": f"PyTorch {safe_slug}",
                "dims": dims_nx_ny_nz(state),
                "cells_file": f"cells/{safe_slug}.f32",
                "params": params.to_catalog_dict(),
            }
        )

    manifest = {
        "format_version": 1,
        "layout": "x-fastest",
        "source": source,
        "animals": animals,
    }
    manifest_path = out_dir / "catalog.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_single_state_catalog(
    state: torch.Tensor,
    out_dir: Path,
    params: LeniaParams,
    *,
    slug: str = "torch_state",
    target: torch.Tensor | None = None,
) -> Path:
    summary = state_summary(state, target)
    return write_catalog({slug: state}, out_dir, params, metrics=summary)


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        for key in ("state", "A", "tensor"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key]
    raise ValueError(f"Could not find a tensor in {path}")
