from __future__ import annotations

import json
from pathlib import Path

import torch

from vollenia_model.expanded_flow import ModelSpecSimulator, seed_state
from vollenia_model.export_state import export_model_state
from vollenia_model.flow_transport import reintegrate_sigma_half, transport_dd
from vollenia_model.growth import growth
from vollenia_model.kernels import build_kernel_bank
from vollenia_model.spec import GrowthSpec, load_model_spec


ROOT = Path(__file__).resolve().parents[2]


def test_modelspec_parse_flow_three_channel() -> None:
    spec = load_model_spec(ROOT / "configs/modelspec/flow_three_channel_complex.json")
    assert spec.name == "flow_three_channel_complex"
    assert spec.channel_count == 3
    assert spec.kernel_count == 7
    assert spec.update_mode == "flow"


def test_kernel_bank_normalized_and_finite() -> None:
    spec = load_model_spec(ROOT / "configs/modelspec/flow_three_channel_complex.json")
    bank = build_kernel_bank(spec, dims=(16, 16, 16), device="cuda")
    assert bank.spatial.shape == (spec.kernel_count, 16, 16, 16)
    assert torch.isfinite(bank.spatial).all()
    assert torch.allclose(bank.spatial.sum(dim=(-3, -2, -1)), torch.ones(spec.kernel_count, device="cuda"), atol=1.0e-4)


def test_growth_families_are_finite() -> None:
    u = torch.linspace(0.0, 1.0, 32, device="cuda")
    for family in ("gaussian", "polynomial_lenia3d"):
        value = growth(u, GrowthSpec(family=family, mu=0.12, sigma=0.04))
        assert torch.isfinite(value).all()
        assert value.shape == u.shape


def test_expanded_additive_step_shape_and_finite() -> None:
    spec = load_model_spec(ROOT / "configs/modelspec/expanded_single_kernel.json")
    sim = ModelSpecSimulator(spec, dims=(16, 16, 16), device="cuda")
    state = seed_state(spec, dims=(16, 16, 16), device="cuda")
    nxt = sim.step(state)
    assert nxt.shape == state.shape
    assert torch.isfinite(nxt).all()
    assert torch.all(nxt >= 0.0)


def test_flow_step_shape_nonnegative_and_finite() -> None:
    spec = load_model_spec(ROOT / "configs/modelspec/flow_single_channel.json")
    sim = ModelSpecSimulator(spec, dims=(16, 16, 16), device="cuda")
    state = seed_state(spec, dims=(16, 16, 16), device="cuda")
    nxt = sim.step(state)
    assert nxt.shape == state.shape
    assert torch.isfinite(nxt).all()
    assert torch.all(nxt >= 0.0)


def test_reintegration_sigma_half_mass_conservation_zero_flow() -> None:
    state = torch.rand((8, 8, 8), device="cuda")
    flow = torch.zeros((3, 8, 8, 8), device="cuda")
    nxt = reintegrate_sigma_half(state, flow, dt=0.14, flow_max=1.0, reintegration_dd=1)
    assert torch.allclose(nxt.sum(), state.sum(), atol=1.0e-4, rtol=1.0e-5)
    assert transport_dd(0.14, 1.0, 0.5, 0) == 1


def test_model_state_export_manifest_and_size(tmp_path) -> None:
    spec_path = ROOT / "configs/modelspec/flow_two_channel.json"
    spec = load_model_spec(spec_path)
    state = torch.zeros((2, 4, 5, 6), device="cuda")
    manifest_path = export_model_state(state, tmp_path, spec, model_spec_path=spec_path)
    root = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert root["layout"] == "channel-major-x-fastest"
    assert root["dims"] == [6, 5, 4]
    assert root["channels"] == 2
    state_path = tmp_path / root["state_file"]
    assert state_path.stat().st_size == state.numel() * 4
