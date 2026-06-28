from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .spec import ModelSpec


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")
    except ValueError:
        return str(path)


def export_model_state(
    state: torch.Tensor,
    out_dir: str | Path,
    spec: ModelSpec,
    *,
    model_spec_path: str | Path | None = None,
    manifest_name: str = "state.json",
    state_name: str = "state.f32",
    composite: bool = True,
    render_channel: int | None = None,
) -> Path:
    if state.ndim != 4:
        raise ValueError(f"Expected [C,Z,Y,X] state, got {tuple(state.shape)}")
    c, nz, ny, nx = state.shape
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    state_path = out_path / state_name
    array = state.detach().contiguous().cpu().numpy().astype("<f4", copy=False)
    array.tofile(state_path)

    spec_path = Path(model_spec_path) if model_spec_path is not None else spec.source_path
    if spec_path is None:
        spec_ref = ""
    else:
        spec_ref = _relative_or_absolute(Path(spec_path), out_path)
    manifest = {
        "format_version": 1,
        "model_spec": spec_ref,
        "layout": "channel-major-x-fastest",
        "dims": [int(nx), int(ny), int(nz)],
        "channels": int(c),
        "state_file": state_path.name,
        "render": {
            "composite": bool(composite),
            "render_channel": int(spec.render_channel if render_channel is None else render_channel),
        },
    }
    manifest_path = out_path / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
