from __future__ import annotations

import json

import torch

from vollenia_diff.export_cpp import apply_export_activation, write_single_state_catalog
from vollenia_diff.params import LeniaParams


def test_export_manifest_references_expected_f32(tmp_path) -> None:
    state = torch.arange(4 * 5 * 6, dtype=torch.float32, device="cuda").reshape(4, 5, 6)
    manifest_path = write_single_state_catalog(
        state,
        tmp_path,
        LeniaParams(),
        slug="unit_state",
        simulation_dims=[8, 8, 8],
        resolution_policy="cropped",
        animal_metadata={"goal_profile": "unit"},
    )
    root = json.loads(manifest_path.read_text(encoding="utf-8"))
    animal = root["animals"][0]
    assert root["layout"] == "x-fastest"
    assert animal["dims"] == [6, 5, 4]
    assert animal["simulation_dims"] == [8, 8, 8]
    assert animal["resolution_policy"] == "cropped"
    assert animal["goal_profile"] == "unit"
    cells_path = tmp_path / animal["cells_file"]
    assert cells_path.exists()
    assert cells_path.stat().st_size == state.numel() * 4


def test_export_activation_clamp_bounds_renderer_density() -> None:
    state = torch.tensor([-2.0, -0.25, 0.5, 1.75], dtype=torch.float32, device="cuda")
    activated = apply_export_activation(state, "clamp")
    assert torch.all(activated >= 0.0)
    assert torch.all(activated <= 1.0)
    assert activated.tolist() == [0.0, 0.0, 0.5, 1.0]
