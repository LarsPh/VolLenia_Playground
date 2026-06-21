from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch

from vollenia_diff.io import load_catalog_animal_state
from vollenia_diff.metrics import normalized_descriptors
from vollenia_diff.params import LeniaParams
from vollenia_diff.rollout_losses import balanced_target_loss, maintain_animal_profile_loss, move_shape_target_loss, rollout_collect, target_sphere_mask
from vollenia_diff.simulator import LeniaSimulator, make_seed_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_sensorimotor_mvp import (
    _apply_rule_noise,
    candidate_mutation_decision,
    evaluate_life_gate,
    gradient_diagnostics,
    make_optimizer,
    parse_args,
    procedural_source_state,
    ranked_entries,
    score_100_from_loss,
)


def test_straight_through_hard_is_bounded_and_grad_nonzero() -> None:
    shape = (12, 12, 12)
    simulator = LeniaSimulator(shape, LeniaParams(R=4.0), device="cuda", clip_mode="straight_through_hard")
    state = make_seed_state(shape, device="cuda", seed=13)
    logits = torch.logit(torch.clamp(state, 1.0e-4, 1.0 - 1.0e-4)).detach().requires_grad_(True)
    final = torch.sigmoid(logits)
    for _ in range(3):
        final = simulator.step(final)
    loss = final.square().mean()
    loss.backward()
    assert torch.isfinite(final).all()
    assert final.min() >= 0.0
    assert final.max() <= 1.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm() > 0.0


def test_normalized_descriptors_and_profile_losses_are_finite() -> None:
    shape = (12, 12, 12)
    simulator = LeniaSimulator(shape, LeniaParams(R=4.0), device="cuda", clip_mode="straight_through_hard")
    initial = make_seed_state(shape, device="cuda", seed=19)
    states = rollout_collect(simulator, initial, 3, sample_interval=1)
    target = torch.tensor([6.0, 6.0, 7.5], device="cuda")
    descriptors = normalized_descriptors(states[-1], initial=initial, target=target)
    for value in descriptors.values():
        assert torch.isfinite(value).all()
    move_loss, move_terms = move_shape_target_loss(states, target=target)
    configured_loss, configured_terms = move_shape_target_loss(
        states,
        target=target,
        objective={
            "weights": {"active_ratio": 30.0},
            "windows": {"active_ratio": [0.9, 1.1]},
        },
    )
    maintain_loss, maintain_terms = maintain_animal_profile_loss(states)
    assert torch.isfinite(move_loss)
    assert torch.isfinite(configured_loss)
    assert torch.isfinite(maintain_loss)
    assert torch.isfinite(move_terms["loss_total"])
    assert torch.isfinite(configured_terms["active_ratio"])
    assert "active_ratio" in move_terms
    assert "absolute_occupancy" in move_terms
    assert "balanced_target" in move_terms
    assert torch.isfinite(maintain_terms["loss_total"])


def test_score_100_and_life_gate_classification() -> None:
    assert score_100_from_loss(0.0) == 100.0
    assert score_100_from_loss(9.0) == 10.0
    assert score_100_from_loss(1.0) > score_100_from_loss(2.0)
    gates = {
        "eval_gates": {
            "min_max_density": 0.08,
            "mass_fraction": [0.0001, 0.06],
            "active_fraction": [0.0005, 0.12],
            "body_radius_norm": [0.02, 0.28],
            "max_border_mass": 0.12,
        }
    }
    dead = evaluate_life_gate(
        {"max_density": 0.0, "mass_fraction": 0.0, "active_fraction": 0.0, "body_radius_norm": 0.0, "border_mass": 0.0},
        gates,
    )
    assert dead["life_gate_pass"] is False
    assert dead["collapse_reason"] == "dead_density"
    noise = evaluate_life_gate(
        {"max_density": 1.0, "mass_fraction": 1.0, "active_fraction": 1.0, "body_radius_norm": 0.35, "border_mass": 0.2},
        gates,
    )
    assert noise["life_gate_pass"] is False
    assert noise["collapse_reason"] in {"over_active_noise", "over_mass_noise", "too_diffuse", "border_leak"}
    ok = evaluate_life_gate(
        {"max_density": 0.5, "mass_fraction": 0.01, "active_fraction": 0.02, "body_radius_norm": 0.1, "border_mass": 0.01},
        gates,
    )
    assert ok["life_gate_pass"] is True
    assert ok["collapse_reason"] == "ok"


def test_balanced_target_penalizes_global_activation() -> None:
    target = torch.tensor([8.0, 8.0, 8.0], device="cuda")
    mask = target_sphere_mask((16, 16, 16), target, radius=3.0)
    focused = mask * 0.8
    global_active = torch.ones((16, 16, 16), device="cuda")
    assert balanced_target_loss(global_active, mask, 0.8) > balanced_target_loss(focused, mask, 0.8)


def test_ranked_entries_use_rank_score_100() -> None:
    ranked = ranked_entries([
        {"id": "bad", "score": 90.0, "rank_score_100": 5.0},
        {"id": "good", "score": 30.0, "rank_score_100": 30.0},
    ])
    assert ranked[0]["id"] == "good"


def test_catalog_loader_center_pads_cropped_state(tmp_path) -> None:
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    cells = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
    cells.numpy().astype("<f4").tofile(cells_dir / "unit.f32")
    manifest = {
        "format_version": 1,
        "layout": "x-fastest",
        "animals": [
            {
                "id": 0,
                "slug": "unit",
                "dims": [4, 4, 4],
                "simulation_dims": [8, 8, 8],
                "resolution_policy": "cropped",
                "cells_file": "cells/unit.f32",
                "params": LeniaParams().to_catalog_dict(),
            }
        ],
    }
    manifest_path = tmp_path / "catalog.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = load_catalog_animal_state(manifest_path, "unit", device="cuda")
    assert tuple(source.state.shape) == (8, 8, 8)
    assert torch.allclose(source.state[2:6, 2:6, 2:6].cpu(), cells)
    assert source.source_metadata["simulation_dims"] == [8, 8, 8]


def test_catalog_loader_resamples_and_scales_radius(tmp_path) -> None:
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    cells = torch.ones((4, 4, 4), dtype=torch.float32)
    cells.numpy().astype("<f4").tofile(cells_dir / "unit.f32")
    manifest = {
        "format_version": 1,
        "layout": "x-fastest",
        "animals": [
            {
                "id": 0,
                "slug": "unit",
                "dims": [4, 4, 4],
                "simulation_dims": [4, 4, 4],
                "resolution_policy": "native",
                "cells_file": "cells/unit.f32",
                "params": LeniaParams(R=5.0).to_catalog_dict(),
            }
        ],
    }
    manifest_path = tmp_path / "catalog.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = load_catalog_animal_state(manifest_path, "unit", size=8, device="cuda")
    assert tuple(source.state.shape) == (8, 8, 8)
    assert source.params.R == 10.0
    assert source.source_metadata["rule_radius_scale"] == 2.0


def test_procedural_source_resamples_and_scales_radius() -> None:
    state, params, metadata = procedural_source_state(32, 16, 123, torch.device("cuda"))
    assert tuple(state.shape) == (32, 32, 32)
    assert params.R == 20.0
    assert metadata["source_size"] == 16
    assert metadata["rule_radius_scale"] == 2.0


def test_procedural_source_rule_randomization_is_legal() -> None:
    state, params, metadata = procedural_source_state(
        32,
        16,
        123,
        torch.device("cuda"),
        {
            "enabled": True,
            "R_log_std": 0.2,
            "m_std": 0.05,
            "s_log_std": 0.2,
            "b_std": 0.1,
            "T_log_std": 0.0,
        },
    )
    assert tuple(state.shape) == (32, 32, 32)
    assert params.R != 20.0
    assert 0.0 < params.m < 0.5
    assert params.s > 0.0
    assert params.T == 10.0
    assert all(0.0 <= value <= 1.0 for value in params.b)
    assert metadata["rule_randomization"]["enabled"] is True


def test_procedural_source_rule_randomization_can_change_ring_count() -> None:
    _, params, metadata = procedural_source_state(
        32,
        16,
        321,
        torch.device("cuda"),
        {
            "enabled": True,
            "probability": 1.0,
            "ring_count_probability": 1.0,
            "ring_count_choices": [2],
            "kn_probability": 1.0,
            "kn_choices": [3],
            "gn_probability": 1.0,
            "gn_choices": [2],
            "R_log_std": 0.0,
            "m_std": 0.0,
            "s_log_std": 0.0,
            "b_std": 0.0,
            "T_log_std": 0.0,
        },
    )
    assert len(params.b) == 2
    assert params.kn == 3
    assert params.gn == 2
    assert metadata["rule_randomization"]["fields"]["ring_count"]["after"] == 2


def test_rule_mutation_returns_candidate_copy() -> None:
    base = LeniaParams(R=10.0, m=0.12, s=0.01, b=[1.0, 0.75])
    before = base.to_catalog_dict()
    mutated, metadata = _apply_rule_noise(
        base,
        {
            "enabled": True,
            "R_log_std": 0.2,
            "m_std": 0.05,
            "s_log_std": 0.2,
            "b_std": 0.1,
            "T_log_std": 0.0,
        },
        seed=7,
        label="candidate_mutation",
    )
    assert base.to_catalog_dict() == before
    assert mutated.to_catalog_dict() != before
    assert metadata["label"] == "candidate_mutation"


def test_rule_mutation_probability_and_b_element_mask() -> None:
    base = LeniaParams(R=10.0, m=0.12, s=0.01, b=[1.0, 0.75, 0.5, 0.25], kn=1)
    unchanged, metadata = _apply_rule_noise(
        base,
        {
            "enabled": True,
            "probability": 0.0,
            "R_log_std": 0.5,
            "m_std": 0.5,
            "s_log_std": 0.5,
            "b_std": 0.5,
        },
        seed=5,
        label="candidate_mutation",
    )
    assert unchanged.to_catalog_dict() == base.to_catalog_dict()
    assert metadata["applied"] is False

    mutated, metadata = _apply_rule_noise(
        base,
        {
            "enabled": True,
            "probability": 1.0,
            "R_probability": 0.0,
            "R_log_std": 0.5,
            "m_probability": 0.0,
            "m_std": 0.5,
            "s_probability": 0.0,
            "s_log_std": 0.5,
            "b_probability": 1.0,
            "b_element_probability": 1.0,
            "b_std": 0.5,
        },
        seed=8,
        label="candidate_mutation",
    )
    assert mutated.R == base.R
    assert mutated.m == base.m
    assert mutated.s == base.s
    assert len(mutated.b) == len(base.b)
    assert mutated.b != base.b
    assert all(metadata["fields"]["b"]["mask"])


def test_candidate_mutation_schedule_short_and_full_steps() -> None:
    mutation = {
        "probability": 1.0,
        "no_mutation_every_iterations": 5,
        "mutated_inner_optim_steps": 7,
        "initial_logit_probability": 1.0,
        "initial_logit_std": 0.02,
        "rule": {"enabled": True, "probability": 1.0, "m_std": 0.01},
    }
    effective, metadata, inner_steps = candidate_mutation_decision(mutation, iteration=0, seed=10, full_inner_steps=20)
    assert metadata["applied"] is True
    assert inner_steps == 7
    assert effective["initial_logit_std"] == 0.02
    assert effective["rule"]["enabled"] is True

    effective, metadata, inner_steps = candidate_mutation_decision(mutation, iteration=4, seed=10, full_inner_steps=20)
    assert metadata["applied"] is False
    assert metadata["reason"] == "scheduled_no_mutation"
    assert inner_steps == 20
    assert effective["initial_logit_std"] == 0.0
    assert effective["rule"]["enabled"] is False

    no_mutation = {**mutation, "probability": 0.0}
    effective, metadata, inner_steps = candidate_mutation_decision(no_mutation, iteration=0, seed=10, full_inner_steps=20)
    assert metadata["applied"] is False
    assert metadata["reason"] == "mutation_probability"
    assert inner_steps == 20

    long_mutation = {**mutation, "mutated_inner_optim_steps": 50}
    _, metadata, inner_steps = candidate_mutation_decision(long_mutation, iteration=0, seed=10, full_inner_steps=20)
    assert metadata["applied"] is True
    assert inner_steps == 20


def test_optimizer_default_lr_and_gradient_diagnostics() -> None:
    logits = torch.zeros((4, 4, 4), device="cuda", requires_grad=True)
    raw_m = torch.zeros((), device="cuda", requires_grad=True)
    args = Namespace(
        optimizer_name="Adam",
        optimizer_lr=0.08,
        optimizer_initial_lr=None,
        optimizer_param_lr=None,
        max_grad_norm=None,
    )
    optimizer = make_optimizer(args, logits, {"m": raw_m})
    assert optimizer.param_groups[0]["lr"] == 0.08
    loss = (torch.sigmoid(logits).mean() + raw_m.square())
    loss.backward()
    diagnostics = gradient_diagnostics(logits, {"m": raw_m}, max_grad_norm=None)
    assert diagnostics["grad_norm_global"] > 0.0
    assert diagnostics["logit_grad_rms"] > 0.0
    assert "m_grad" in diagnostics


def test_config_load_and_cli_override(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "search.yaml"
    out_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "profile: move_shape_target",
                "simulation:",
                "  size: 64",
                "  steps: 10",
                "search:",
                "  iterations: 9",
                "  mutation:",
                "    initial_logit_std: 0.02",
                "    rule:",
                "      enabled: true",
                "      m_std: 0.01",
                "optimization:",
                "  inner_optim_steps: 7",
                "  optimize_params: [m, s]",
                "  optimizer:",
                "    lr: 0.08",
                "objective:",
                "  target_offset_norm_zyx: [0.0, 0.0, 0.2]",
                "  weights:",
                "    active_ratio: 12.0",
                "export:",
                f"  out: {out_dir}",
                "  save_initial_states: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_sensorimotor_mvp.py",
            "--config",
            str(config_path),
            "--iterations",
            "3",
            "--size",
            "32",
            "--debug-allow-size-32",
            "--mutation-std",
            "0.05",
        ],
    )
    args = parse_args()
    assert args.profile == "move_shape_target"
    assert args.steps == 10
    assert args.iterations == 3
    assert args.size == 32
    assert args.out == out_dir
    assert args.save_initial_states is True
    assert args.optimize_params == "m,s"
    assert args.objective["target_offset_norm_zyx"] == [0.0, 0.0, 0.2]
    assert args.mutation["initial_logit_std"] == 0.05
    assert args.mutation["rule"]["enabled"] is True


def test_script_default_does_not_optimize_T(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_sensorimotor_mvp.py",
            "--profile",
            "move_shape_target",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    args = parse_args()
    assert args.optimize_params == "m,s"


def test_search_script_debug_size_32_smoke(tmp_path) -> None:
    out_dir = tmp_path / "search"
    result = subprocess.run(
        [
            sys.executable,
            "python/scripts/search_sensorimotor_mvp.py",
            "--profile",
            "move_shape_target",
            "--size",
            "32",
            "--debug-allow-size-32",
            "--steps",
            "2",
            "--iterations",
            "1",
            "--inner-optim-steps",
            "1",
            "--train-clip-mode",
            "straight_through_hard",
            "--out",
            str(out_dir),
            "--checkpoint-every-iterations",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote search outputs" in result.stdout
    root = json.loads((out_dir / "catalog.json").read_text(encoding="utf-8"))
    animal = root["animals"][0]
    assert animal["simulation_dims"] == [32, 32, 32]
    assert animal["resolution_policy"] == "native"
    assert (out_dir / animal["cells_file"]).stat().st_size == 32 * 32 * 32 * 4
    initial_path = out_dir / "initials" / "move_shape_target_0000_initial.f32"
    assert initial_path.exists()
    assert initial_path.stat().st_size == 32 * 32 * 32 * 4
    assert "Debug-only run" in (out_dir / "summary.md").read_text(encoding="utf-8")
    assert (out_dir / "best" / "catalog.json").exists()
    best_manifest = json.loads((out_dir / "best" / "catalog.json").read_text(encoding="utf-8"))
    assert best_manifest["animals"][0]["learned_initial_file"] == "initials/move_shape_target_0000_initial.f32"
    assert (out_dir / "best" / "initials" / "move_shape_target_0000_initial.f32").exists()
    assert (out_dir / "initial_catalog" / "catalog.json").exists()
    initial_manifest = json.loads((out_dir / "initial_catalog" / "catalog.json").read_text(encoding="utf-8"))
    assert initial_manifest["animals"][0]["state_role"] == "learned_initial"
    assert (out_dir / "checkpoints" / "iter_00001" / "catalog.json").exists()
    entry = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["entries"][0]
    assert "gradient_diagnostics" in entry
    assert "logit_grad_rms" in entry["gradient_diagnostics"]
    assert "mutation_decision" in entry
    assert entry["inner_steps_used"] == 1
    assert "score_100" in entry
    assert "rank_score_100" in entry
    assert "loss_score" in entry
    assert "life_gate_pass" in entry
    csv_text = (out_dir / "candidates.csv").read_text(encoding="utf-8")
    assert "rank_score_100" in csv_text
    assert "collapse_reason" in csv_text
