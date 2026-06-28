from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from vollenia_diff.io import load_catalog_animal_state
from vollenia_diff.metrics import descriptors_to_json, normalized_descriptors, soft_active_fraction, soft_active_ratio, state_summary
from vollenia_diff.params import LeniaParams
from vollenia_diff.rollout_losses import (
    PROFILE_LOSSES,
    balanced_target_loss,
    move_shape_target_loss,
    rescue_unstable_animal_loss,
    rollout_collect,
    target_sphere_mask,
)
from vollenia_diff.simulator import LeniaSimulator, make_seed_state
from vollenia_search.archive import choose_source, ranked_entries, score_100_from_loss
from vollenia_search.config import _config_to_args, _load_yaml_config, load_search_config, parse_args
from vollenia_search.evaluate import _continuation_horizons, evaluate_candidate, evaluate_continuation_gates, evaluate_life_gate
from vollenia_search.mutation import _apply_rule_noise, candidate_mutation_decision, precheck_mutation_candidate
from vollenia_search.optimize import gradient_diagnostics, make_optimizer
from vollenia_search.sources import inject_procedural_source, procedural_source_state, should_inject_source
from vollenia_search.targets import make_target_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_sensorimotor_mvp as search_mvp
import vollenia_search.evaluate as search_eval


def _read_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_file(str(path), dtype=torch.float32, size=int(torch.tensor(shape).prod().item())).reshape(shape)


def _topology_objective() -> dict[str, object]:
    return {
        "eval_gates": {
            "min_max_density": 0.08,
            "topology": {
                "enabled": True,
                "allowed": ["blob", "cylinder", "plane"],
                "full_axis_span_min": 0.85,
                "compact_axis_span_max": 0.72,
                "compact_axis_border_mass_max": 0.18,
                "by_type": {
                    "blob": {
                        "mass_fraction": [0.00005, 0.08],
                        "active_fraction": [0.0002, 0.16],
                        "active_body_radius_norm": [0.015, 0.45],
                        "max_border_mass": 0.15,
                    },
                    "cylinder": {
                        "mass_fraction": [0.00005, 0.14],
                        "active_fraction": [0.0002, 0.24],
                        "compact_axis_span_norm": [0.01, 0.55],
                        "compact_axis_border_mass_max": 0.18,
                    },
                    "plane": {
                        "mass_fraction": [0.00005, 0.18],
                        "active_fraction": [0.0002, 0.35],
                        "compact_axis_span_norm": [0.01, 0.72],
                        "compact_axis_border_mass_max": 0.18,
                    },
                },
            },
        }
    }


def _topology_gate_for_state(state: torch.Tensor, objective: dict[str, object] | None = None) -> dict[str, object]:
    desc = descriptors_to_json(normalized_descriptors(state))
    summary = state_summary(state)
    return evaluate_life_gate({**summary, **desc}, objective or _topology_objective())


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
    assert "active_body_radius_norm" in descriptors
    move_loss, move_terms = move_shape_target_loss(states, target=target)
    configured_loss, configured_terms = move_shape_target_loss(
        states,
        target=target,
        objective={
            "weights": {"active_ratio": 30.0},
            "windows": {"active_ratio": [0.9, 1.1]},
        },
    )
    rescue_loss, rescue_terms = rescue_unstable_animal_loss(states)
    assert torch.isfinite(move_loss)
    assert torch.isfinite(configured_loss)
    assert torch.isfinite(rescue_loss)
    assert torch.isfinite(move_terms["loss_total"])
    assert torch.isfinite(configured_terms["active_ratio"])
    assert "active_ratio" in move_terms
    assert "absolute_occupancy" in move_terms
    assert "balanced_target" in move_terms
    assert torch.isfinite(rescue_terms["loss_total"])
    assert "absolute_occupancy" in rescue_terms


def test_soft_active_terms_have_gradients_near_threshold() -> None:
    state = torch.full((4, 4, 4), 0.05, device="cuda", requires_grad=True)
    initial = torch.full((4, 4, 4), 0.04, device="cuda")
    loss = soft_active_fraction(state) + soft_active_ratio(state, initial)
    loss.backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0.0


def test_rollout_collect_returns_final_step_for_sparse_sampling() -> None:
    simulator = LeniaSimulator((8, 8, 8), LeniaParams(R=3.0), device="cuda", clip_mode="hard")
    initial = make_seed_state((8, 8, 8), device="cuda", seed=21)
    states, steps = rollout_collect(simulator, initial, 5, sample_interval=2, return_steps=True)
    assert len(states) == len(steps)
    assert steps == [0, 2, 4, 5]


def test_staged_move_target_profile_adds_stage_terms() -> None:
    shape = (12, 12, 12)
    simulator = LeniaSimulator(shape, LeniaParams(R=4.0), device="cuda", clip_mode="straight_through_hard")
    initial = make_seed_state(shape, device="cuda", seed=22)
    states, steps = rollout_collect(simulator, initial, 4, sample_interval=2, return_steps=True)
    target = torch.tensor([6.0, 6.0, 7.5], device="cuda")
    single_loss, single_terms = move_shape_target_loss(
        states,
        target=target,
        objective={"target_profile": {"mode": "staged_offsets", "stages": [{"at_step_fraction": 1.0, "offset_norm_zyx": [0.0, 0.0, 0.12]}]}},
        state_steps=steps,
        total_steps=4,
    )
    multi_loss, multi_terms = move_shape_target_loss(
        states,
        target=target,
        objective={
            "target_profile": {
                "mode": "staged_offsets",
                "stages": [
                    {"at_step_fraction": 0.5, "offset_norm_zyx": [0.0, 0.0, 0.06], "weight_scale": {"com": 0.35, "balanced_target": 0.35}},
                    {"at_step_fraction": 1.0, "offset_norm_zyx": [0.0, 0.0, 0.12]},
                ],
            }
        },
        state_steps=steps,
        total_steps=4,
    )
    assert torch.isfinite(single_loss)
    assert torch.isfinite(multi_loss)
    assert torch.isfinite(single_terms["stage_0_com"])
    assert torch.isfinite(multi_terms["stage_1_balanced_target"])
    assert int(multi_terms["stage_0_sampled_step"].detach().cpu()) == 2
    assert "stage_1_target_distance_norm" in multi_terms


def test_target_context_is_per_candidate_for_initial_offset() -> None:
    first = torch.zeros((16, 16, 16), device="cuda")
    first[4:6, 4:6, 4:6] = 1.0
    second = torch.zeros((16, 16, 16), device="cuda")
    second[10:12, 10:12, 10:12] = 1.0
    objective = {
        "target_offset_norm_zyx": [0.0, 0.0, 0.1],
        "target_profile": {"mode": "staged_offsets", "stages": [{"at_step_fraction": 1.0, "offset_norm_zyx": [0.0, 0.0, 0.1]}]},
    }
    first_context = make_target_context("move_shape_target", first, objective)
    second_context = make_target_context("move_shape_target", second, objective)
    assert first_context.target is not None
    assert second_context.target is not None
    assert not torch.allclose(first_context.target, second_context.target)
    assert first_context.stages[0].target_zyx != second_context.stages[0].target_zyx


def test_score_100_and_life_gate_classification() -> None:
    assert score_100_from_loss(0.0) == 100.0
    assert score_100_from_loss(9.0) == 10.0
    assert score_100_from_loss(1.0) > score_100_from_loss(2.0)
    gates = {
        "eval_gates": {
            "min_max_density": 0.08,
            "mass_fraction": [0.0001, 0.06],
            "active_fraction": [0.0005, 0.12],
            "active_body_radius_norm": [0.02, 0.42],
            "max_border_mass": 0.12,
        }
    }
    dead = evaluate_life_gate(
        {"max_density": 0.0, "mass_fraction": 0.0, "active_fraction": 0.0, "active_body_radius_norm": 0.0, "border_mass": 0.0},
        gates,
    )
    assert dead["life_gate_pass"] is False
    assert dead["collapse_reason"] == "dead_density"
    noise = evaluate_life_gate(
        {"max_density": 1.0, "mass_fraction": 1.0, "active_fraction": 1.0, "active_body_radius_norm": 0.5, "border_mass": 0.2},
        gates,
    )
    assert noise["life_gate_pass"] is False
    assert noise["collapse_reason"] in {"over_active_noise", "over_mass_noise", "active_too_diffuse_noise", "border_leak"}
    ok = evaluate_life_gate(
        {"max_density": 0.5, "mass_fraction": 0.01, "active_fraction": 0.02, "active_body_radius_norm": 0.1, "border_mass": 0.01},
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


def test_active_body_radius_detects_full_grid_noise() -> None:
    local = torch.zeros((16, 16, 16), device="cuda")
    local[6:10, 6:10, 6:10] = 1.0
    full = torch.ones((16, 16, 16), device="cuda")
    local_desc = normalized_descriptors(local)
    full_desc = normalized_descriptors(full)
    assert torch.isfinite(local_desc["active_body_radius_norm"])
    assert torch.isfinite(full_desc["active_body_radius_norm"])
    assert full_desc["active_body_radius_norm"] > local_desc["active_body_radius_norm"]
    gate = evaluate_life_gate(
        {
            "max_density": 1.0,
            "mass_fraction": 0.03,
            "active_fraction": 0.05,
            "active_body_radius_norm": float(full_desc["active_body_radius_norm"].detach().cpu()),
            "border_mass": 0.0,
        },
        {"eval_gates": {"active_body_radius_norm": [0.02, 0.42]}},
    )
    assert gate["life_gate_pass"] is False
    assert gate["collapse_reason"] == "active_too_diffuse_noise"


def test_topology_gate_classifies_blob_cylinder_plane_and_global_noise() -> None:
    blob = torch.zeros((32, 32, 32), device="cuda")
    blob[12:20, 12:20, 12:20] = 1.0
    blob_gate = _topology_gate_for_state(blob)
    assert blob_gate["life_gate_pass"] is True
    assert blob_gate["life_topology"] == "blob"
    assert blob_gate["full_axis_count"] == 0

    cylinder = torch.zeros((32, 32, 32), device="cuda")
    cylinder[:, 12:20, 12:20] = 1.0
    cylinder_gate = _topology_gate_for_state(cylinder)
    assert cylinder_gate["life_gate_pass"] is True
    assert cylinder_gate["life_topology"] == "cylinder"
    assert cylinder_gate["full_axis_count"] == 1

    plane = torch.zeros((32, 32, 32), device="cuda")
    plane[:, :, 12:20] = 0.5
    plane_gate = _topology_gate_for_state(plane)
    assert plane_gate["life_gate_pass"] is True
    assert plane_gate["life_topology"] == "plane"
    assert plane_gate["full_axis_count"] == 2

    global_noise = torch.ones((32, 32, 32), device="cuda") * 0.2
    noise_gate = _topology_gate_for_state(global_noise)
    assert noise_gate["life_gate_pass"] is False
    assert noise_gate["life_topology"] == "global_noise"
    assert noise_gate["collapse_reason"] == "all_axes_full_noise"


def test_topology_gate_handles_wrapped_blob_and_allowed_filter() -> None:
    wrapped = torch.zeros((32, 32, 32), device="cuda")
    wrapped[:3, 14:18, 14:18] = 1.0
    wrapped[-3:, 14:18, 14:18] = 1.0
    desc = descriptors_to_json(normalized_descriptors(wrapped))
    assert desc["active_axis_circular_span_norm"][0] < 0.25
    gate = _topology_gate_for_state(
        wrapped,
        {
            "eval_gates": {
                "topology": {
                    "enabled": True,
                    "allowed": ["blob", "cylinder", "plane"],
                    "full_axis_span_min": 0.85,
                    "by_type": {"blob": {"mass_fraction": [0.0, 1.0], "active_fraction": [0.0, 1.0]}},
                }
            }
        },
    )
    assert gate["life_topology"] == "blob"

    cylinder = torch.zeros((32, 32, 32), device="cuda")
    cylinder[:, 12:20, 12:20] = 1.0
    filtered = _topology_gate_for_state(
        cylinder,
        {
            "eval_gates": {
                "topology": {
                    "enabled": True,
                    "allowed": ["blob"],
                    "full_axis_span_min": 0.85,
                    "by_type": {"cylinder": {"mass_fraction": [0.0, 1.0], "active_fraction": [0.0, 1.0]}},
                }
            }
        },
    )
    assert filtered["life_gate_pass"] is False
    assert filtered["collapse_reason"] == "topology_filtered"


def test_ranked_entries_use_rank_score_100() -> None:
    ranked = ranked_entries([
        {"id": "bad", "score": 90.0, "rank_score_100": 5.0},
        {"id": "good", "score": 30.0, "rank_score_100": 30.0},
    ])
    assert ranked[0]["id"] == "good"


def test_continuation_failure_caps_rank_score(monkeypatch) -> None:
    def fake_continuation(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "continuation_steps": [3],
            "continuation_results": [{"steps": 3, "life_gate_pass": False, "collapse_reason": "over_active_noise"}],
            "continuation_life_gate_pass": False,
            "continuation_collapse_reason": "over_active_noise",
            "continuation_gate_violations": [{"steps": 3, "reason": "over_active_noise"}],
            "first_failure_horizon": 3,
            "longest_passed_horizon": 1,
            "failed_at_final_horizon": True,
            "skipped_continuation_steps": [],
        }

    monkeypatch.setattr(search_eval, "evaluate_continuation_gates", fake_continuation)
    state0 = make_seed_state((12, 12, 12), device="cuda", seed=4)
    _, metrics, terms, score = evaluate_candidate(
        state0,
        LeniaParams(R=4.0),
        profile="move_shape_target",
        target=torch.tensor([6.0, 6.0, 7.0], device="cuda"),
        steps=1,
        clip_mode="hard",
        compile_step=False,
        objective={"gate_fail_score_cap": 5.0},
        evaluation={"continuation_steps": [3]},
    )
    assert metrics["primary_life_gate_pass"] is True
    assert metrics["continuation_life_gate_pass"] is False
    assert metrics["life_gate_pass"] is False
    assert metrics["collapse_reason"] == "continuation_over_active_noise"
    assert metrics["rank_score_100"] == min(metrics["score_100"], 5.0)
    assert score == metrics["rank_score_100"]
    assert terms["continuation_life_gate_pass"] is False
    assert metrics["first_failure_horizon"] == 3
    assert metrics["longest_passed_horizon"] == 1


def test_continuation_horizons_are_absolute_steps() -> None:
    assert _continuation_horizons({"continuation_steps": [5, 10, 3, 10]}, 5) == [10]


def test_continuation_stop_on_first_failure_metadata(monkeypatch) -> None:
    calls = {"count": 0}

    def always_fail(*args: object, **kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        return {
            "life_gate_pass": False,
            "collapse_reason": "over_active_noise",
            "gate_violations": [{"reason": "over_active_noise"}],
            "eval_gates": {},
        }

    monkeypatch.setattr(search_eval, "evaluate_life_gate", always_fail)
    state0 = make_seed_state((8, 8, 8), device="cuda", seed=23)
    params = LeniaParams(R=3.0)
    primary_final = LeniaSimulator((8, 8, 8), params, device="cuda", clip_mode="hard").step(state0)
    result = evaluate_continuation_gates(
        state0,
        primary_final,
        params,
        target=None,
        primary_steps=1,
        clip_mode="hard",
        compile_step=False,
        objective={},
        evaluation={"continuation_steps": [3, 5, 7], "stop_on_first_failure": True},
    )
    assert result["continuation_life_gate_pass"] is False
    assert result["first_failure_horizon"] == 3
    assert result["longest_passed_horizon"] == 1
    assert result["skipped_continuation_steps"] == [5, 7]
    assert calls["count"] == 1


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


def test_procedural_seed_modes_are_finite_and_diverse() -> None:
    shape = (24, 24, 24)
    states = {
        mode: make_seed_state(shape, device="cuda", seed=123, mode=mode)
        for mode in ("blob_shell", "random_patch", "mixed_blobs")
    }
    for state in states.values():
        assert torch.isfinite(state).all()
        assert state.min() >= 0.0
        assert state.max() <= 1.0
        assert state.sum() > 0.0
    assert not torch.allclose(states["blob_shell"], states["random_patch"])
    assert not torch.allclose(states["blob_shell"], states["mixed_blobs"])


def test_procedural_source_initialization_config_metadata() -> None:
    state, _, metadata = procedural_source_state(
        32,
        16,
        123,
        torch.device("cuda"),
        initialization={"mode": "random_patch", "patch_size_fraction": [0.5, 0.5]},
    )
    assert tuple(state.shape) == (32, 32, 32)
    assert metadata["initialization"]["mode"] == "random_patch"


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


def test_rescue_source_rule_randomization_changes_rms_but_not_structure() -> None:
    config = _load_yaml_config(Path("configs/search_mvp/rescue_unstable_animal/default.yaml"))
    args = _config_to_args(config)
    base = LeniaParams(R=10.0, m=0.12, s=0.01, T=10.0, b=[1.0, 0.5, 0.25], kn=1, gn=1)
    mutated, metadata = _apply_rule_noise(
        base,
        args["rule_randomization"],
        seed=12,
        label="source_rule_randomization",
        allow_structure=True,
    )
    assert mutated.R != base.R
    assert mutated.m != base.m
    assert mutated.s != base.s
    assert mutated.T == base.T
    assert mutated.kn == base.kn
    assert mutated.gn == base.gn
    assert len(mutated.b) == len(base.b)
    assert mutated.b == base.b
    assert metadata["label"] == "source_rule_randomization"


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


def test_top_alive_weighted_selection_prefers_alive_after_fresh_pool() -> None:
    fresh = {"entry_id": "fresh_0", "score": float("-inf"), "used": False}
    dead = {"entry_id": "dead_0", "score": 90.0, "rank_score_100": 5.0, "life_gate_pass": False}
    alive_low = {"entry_id": "alive_low", "score": 10.0, "rank_score_100": 10.0, "life_gate_pass": True}
    alive_high = {"entry_id": "alive_high", "score": 30.0, "rank_score_100": 30.0, "life_gate_pass": True}
    source, metadata = choose_source(
        [fresh, dead, alive_low, alive_high],
        "top_alive_weighted",
        selection_config={"top_k": 2, "temperature": 6.0},
        seed=1,
        gate_fail_score_cap=5.0,
    )
    assert source is fresh
    assert metadata["selected_by"] == "fresh_unscored"

    fresh["used"] = True
    source, metadata = choose_source(
        [fresh, dead, alive_low, alive_high],
        "top_alive_weighted",
        selection_config={"top_k": 2, "temperature": 6.0},
        seed=1,
        gate_fail_score_cap=5.0,
    )
    assert source["entry_id"] in {"alive_low", "alive_high"}
    assert metadata["selected_by"] == "top_alive_weighted"
    assert "dead_0" not in metadata["candidate_ids"]


def test_top_weighted_selection_uses_top_scores_without_life_gate_requirement() -> None:
    dead_high = {"entry_id": "dead_high", "score": -1.0, "score_100": 80.0, "rank_score_100": 5.0, "life_gate_pass": False}
    alive_mid = {"entry_id": "alive_mid", "score": -2.0, "score_100": 40.0, "rank_score_100": 40.0, "life_gate_pass": True}
    dead_low = {"entry_id": "dead_low", "score": -3.0, "score_100": 10.0, "rank_score_100": 5.0, "life_gate_pass": False}
    source, metadata = choose_source(
        [dead_low, alive_mid, dead_high],
        "top_weighted",
        selection_config={"top_k": 2, "temperature": 0.001, "score_field": "score_100"},
        seed=3,
        gate_fail_score_cap=5.0,
    )
    assert source["entry_id"] == "dead_high"
    assert metadata["selected_by"] == "top_weighted"
    assert metadata["score_field"] == "score_100"
    assert "dead_high" in metadata["candidate_ids"]


def test_source_selection_adaptive_and_rescue_mixed_score_fields() -> None:
    dead_high = {
        "entry_id": "dead_high",
        "score": 5.0,
        "score_100": 80.0,
        "rank_score_100": 5.0,
        "life_gate_pass": False,
        "longest_passed_horizon": 50,
    }
    alive_mid = {
        "entry_id": "alive_mid",
        "score": 40.0,
        "score_100": 40.0,
        "rank_score_100": 40.0,
        "life_gate_pass": True,
        "longest_passed_horizon": 500,
    }
    source, metadata = choose_source(
        [dead_high, alive_mid],
        "top_weighted",
        selection_config={"top_k": 2, "temperature": 0.001, "score_field": "adaptive"},
        seed=3,
        gate_fail_score_cap=5.0,
    )
    assert source["entry_id"] == "alive_mid"
    assert metadata["requested_score_field"] == "adaptive"
    assert metadata["selected_source_score_field"] == "rank_score_100"
    assert metadata["selected_source_longest_passed_horizon"] == 500

    source, metadata = choose_source(
        [dead_high, alive_mid],
        "top_weighted",
        selection_config={"top_k": 2, "temperature": 0.001, "score_field": "rescue_mixed_score"},
        seed=3,
        gate_fail_score_cap=5.0,
    )
    assert source["entry_id"] == "alive_mid"
    assert metadata["score_field"] == "rescue_mixed_score"
    assert metadata["selected_source_life_gate_pass"] is True


def test_source_injection_schedule_and_metadata() -> None:
    args = Namespace(
        source_injection={"enabled": True, "every_iterations": 10, "start_after_initial_pool": True, "max_injections": 2},
        random_init_count=40,
        seed=123,
        size=16,
        source_size=8,
        rule_randomization={"enabled": False},
        source_initialization={"mode": "random_patch", "patch_size_fraction": [0.5, 0.5]},
    )
    assert should_inject_source(args, iteration=39, injections_done=0) is False
    assert should_inject_source(args, iteration=40, injections_done=0) is True
    assert should_inject_source(args, iteration=45, injections_done=0) is False
    archive: list[dict[str, object]] = []
    item = inject_procedural_source(args, archive, device=torch.device("cuda"), iteration=40, injections_done=0)
    assert item["fresh_injected"] is True
    assert item["score"] == float("-inf")
    assert item["source"]["source_injection"]["iteration"] == 40
    assert tuple(item["state0"].shape) == (16, 16, 16)


def test_mutation_precheck_records_attempts() -> None:
    class DummyProgress:
        def update(self, *args: object, **kwargs: object) -> None:
            return None

    args = Namespace(
        steps=2,
        eval_clip_mode="hard",
        compile_step=False,
        objective={
            "gate_fail_score_cap": 5.0,
            "eval_gates": {
                "min_max_density": 0.0,
                "mass_fraction": [0.0, 1.0],
                "active_fraction": [0.0, 1.0],
                "active_body_radius_norm": [0.0, 1.0],
                "max_border_mass": 1.0,
            },
        },
    )
    source = make_seed_state((8, 8, 8), device="cuda", seed=31)
    seed, metadata = precheck_mutation_candidate(
        source,
        LeniaParams(R=3.0),
        args=args,
        profile="move_shape_target",
        target=torch.tensor([4.0, 4.0, 4.5], device="cuda"),
        mutation={
            "initial_logit_std": 0.01,
            "rule": {"enabled": False},
            "precheck": {"enabled": True, "steps": 1, "max_attempts": 1, "accept_first_life_gate_pass": True},
        },
        mutation_decision={"applied": True},
        mutation_seed=99,
        progress=DummyProgress(),  # type: ignore[arg-type]
        search_task=None,
    )
    assert seed == 99
    assert metadata["attempted"] is True
    assert len(metadata["attempts"]) == 1
    assert "collapse_reason" in metadata["attempts"][0]


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

    grouped_args = Namespace(
        optimizer_name="Adam",
        optimizer_lr=0.08,
        optimizer_initial_lr=0.008,
        optimizer_param_lr=0.0008,
        max_grad_norm=None,
    )
    grouped = make_optimizer(grouped_args, logits, {"m": raw_m})
    assert grouped.param_groups[0]["lr"] == 0.008
    assert grouped.param_groups[1]["lr"] == 0.0008


def test_config_load_and_cli_override(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "search.yaml"
    out_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "profile: move_shape_target",
                "source:",
                "  initialization:",
                "    mode: random_patch",
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
    assert args.source_initialization["mode"] == "random_patch"
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


def test_maintain_profile_is_removed_from_cli_and_loss_registry(tmp_path, monkeypatch) -> None:
    assert set(PROFILE_LOSSES) == {"move_shape_target", "rescue_unstable_animal"}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_sensorimotor_mvp.py",
            "--profile",
            "maintain_animal_profile",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_default_profile_configs_load_expected_search_settings(monkeypatch) -> None:
    cases = [
        ("configs/search_mvp/move_shape_target/default.yaml", 40, 0.008, 0.0008, [0.02, 0.42], "top_weighted", True, 4, [100, 200, 500], "score_100"),
        ("configs/search_mvp/rescue_unstable_animal/default.yaml", 16, 0.002, 0.0005, [0.015, 0.45], "top_weighted", False, 4, [100, 200, 500, 1000], "rank_score_100"),
        ("configs/search_mvp/rescue_unstable_animal/default_animal_24.yaml", 8, 0.002, 0.0005, [0.015, 0.45], "top_weighted", False, 4, [100, 200, 500, 1000, 2000], "score_100"),
    ]
    for case in cases:
        config_path = case[0]
        random_init_count = case[1]
        initial_lr = case[2]
        param_lr = case[3]
        active_radius_window = case[4]
        expected_selection = case[5] if len(case) > 5 else "best"
        expected_injection = case[6] if len(case) > 6 else False
        expected_precheck_attempts = case[7] if len(case) > 7 else 2
        expected_horizons = case[8] if len(case) > 8 else [100, 200, 500]
        expected_score_field = case[9] if len(case) > 9 else "score_100"
        monkeypatch.setattr(sys, "argv", ["search_sensorimotor_mvp.py", "--config", config_path])
        args = parse_args()
        assert args.random_init_count == random_init_count
        assert args.optimizer_initial_lr == initial_lr
        assert args.optimizer_param_lr == param_lr
        assert "mode" in args.source_initialization
        assert args.evaluation["continuation_steps"] == expected_horizons
        assert args.evaluation["stop_on_first_failure"] is True
        assert args.source_selection == expected_selection
        assert args.source_selection_config["top_k"] == 12
        assert args.source_selection_config["score_field"] == expected_score_field
        assert args.source_injection["enabled"] is expected_injection
        assert args.mutation["precheck"]["enabled"] is True
        assert args.mutation["precheck"]["max_attempts"] == expected_precheck_attempts
        if args.profile == "rescue_unstable_animal":
            assert args.rule_randomization["enabled"] is True
            assert args.rule_randomization["R_log_std"] > 0.0
            assert args.rule_randomization["m_std"] > 0.0
            assert args.rule_randomization["s_log_std"] > 0.0
            assert args.rule_randomization["T_log_std"] == 0.0
            assert args.objective["weights"]["absolute_occupancy"] > 0.0
        assert args.objective["eval_gates"]["active_body_radius_norm"] == active_radius_window
        assert args.objective["eval_gates"]["topology"]["enabled"] is True
        assert args.objective["eval_gates"]["topology"]["allowed"] == ["blob", "cylinder", "plane"]
        if args.profile == "move_shape_target":
            assert args.objective["target_profile"]["mode"] == "staged_offsets"
            assert len(args.objective["target_profile"]["stages"]) == 2


def test_dataclass_config_loader_preserves_score_field_defaults() -> None:
    rescue = load_search_config(Path("configs/search_mvp/rescue_unstable_animal/default.yaml"))
    animal24 = load_search_config(Path("configs/search_mvp/rescue_unstable_animal/default_animal_24.yaml"))
    assert rescue.source_selection_config.score_field == "rank_score_100"
    assert animal24.source_selection_config.score_field == "score_100"
    assert rescue.logging.inner_log_every == 5


def _load_rescue_best_gate(suffix: str) -> dict[str, object]:
    root = Path("outputs/search_mvp") / f"rescue_sthard_64_{suffix}" / "best"
    catalog_path = root / "catalog.json"
    if not catalog_path.exists():
        pytest.skip(f"Missing rescue regression checkpoint: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    animal = catalog["animals"][0]
    dims = [int(value) for value in animal["dims"]]
    shape = (dims[2], dims[1], dims[0])
    slug = str(animal["slug"])
    state_path = root / "snapshots" / f"{slug}_final.f32"
    if not state_path.exists():
        state_path = root / str(animal["cells_file"])
    if not state_path.exists():
        pytest.skip(f"Missing rescue regression state: {state_path}")
    state = _read_f32(state_path, shape).to(device="cuda")
    config = _load_yaml_config(Path("configs/search_mvp/rescue_unstable_animal/default.yaml"))
    objective = _config_to_args(config)["objective"]
    gate = _topology_gate_for_state(state, objective)
    gate["run_suffix"] = suffix
    gate["state_path"] = str(state_path)
    return gate


@pytest.mark.parametrize("suffix", ["11", "15", "18"])
def test_rescue_best_blob_like_checkpoints_still_pass_topology_gate(suffix: str) -> None:
    gate = _load_rescue_best_gate(suffix)
    assert gate["life_gate_pass"] is True, gate
    assert gate["life_topology"] in {"blob", "cylinder", "plane"}


@pytest.mark.parametrize("suffix", ["24", "25", "26", "27", "28", "29", "30", "31"])
def test_rescue_best_membrane_checkpoints_pass_as_cylinder_or_plane(suffix: str) -> None:
    gate = _load_rescue_best_gate(suffix)
    assert gate["life_gate_pass"] is True, gate
    assert gate["life_topology"] in {"cylinder", "plane"}, gate
    old_false_reasons = {"over_active_noise", "over_mass_noise", "active_too_diffuse_noise", "border_leak"}
    assert gate["collapse_reason"] not in old_false_reasons


@pytest.mark.parametrize("suffix", ["22"])
def test_rescue_best_known_bad_checkpoints_do_not_pass_topology_gate(suffix: str) -> None:
    gate = _load_rescue_best_gate(suffix)
    assert gate["life_gate_pass"] is False, gate
    if suffix == "24":
        assert gate["collapse_reason"] == "compact_axis_too_wide_noise", gate


def test_search_script_debug_size_32_smoke(tmp_path, monkeypatch, capsys) -> None:
    out_dir = tmp_path / "search"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "search_sensorimotor_mvp.py",
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
    )
    search_mvp.main()
    result = capsys.readouterr()
    assert "Wrote search outputs" in result.out
    root = json.loads((out_dir / "catalog.json").read_text(encoding="utf-8"))
    animal = root["animals"][0]
    assert animal["simulation_dims"] == [32, 32, 32]
    assert animal["resolution_policy"] == "native"
    assert (out_dir / animal["cells_file"]).stat().st_size == 32 * 32 * 32 * 4
    initial_path = out_dir / "initials" / "move_shape_target_0000_initial.f32"
    assert initial_path.exists()
    assert initial_path.stat().st_size == 32 * 32 * 32 * 4
    root_cells_path = out_dir / animal["cells_file"]
    assert root_cells_path.read_bytes() == initial_path.read_bytes()
    assert animal["catalog_state_role"] == "learned_initial"
    assert animal["eval_final_file"] == "snapshots/move_shape_target_0000_final.f32"
    assert "active_body_radius_norm" in animal
    assert "life_topology" in animal
    assert "full_axis_count" in animal
    assert "active_axis_coverage_norm" in animal
    assert "active_axis_circular_span_norm" in animal
    assert "axis_border_mass" in animal
    assert "continuation_life_gate_pass" in animal
    assert "continuation_results" in animal
    assert "first_failure_horizon" in animal
    assert "longest_passed_horizon" in animal
    assert "selection_metadata" in animal
    assert "source_entry_id" in animal
    assert "mutation_precheck" in animal
    snapshot_path = out_dir / animal["eval_final_file"]
    assert snapshot_path.exists()
    assert snapshot_path.stat().st_size == 32 * 32 * 32 * 4
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
    assert entry["mutation_decision"]["reason"] == "fresh_source_no_mutation"
    assert entry["inner_steps_used"] == 1
    assert "score_100" in entry
    assert "rank_score_100" in entry
    assert "loss_score" in entry
    assert "life_gate_pass" in entry
    assert "life_topology" in entry
    assert "full_axis_count" in entry
    assert "first_failure_horizon" in entry
    assert "longest_passed_horizon" in entry
    assert "selection_metadata" in entry
    assert "source_entry_id" in entry
    csv_text = (out_dir / "candidates.csv").read_text(encoding="utf-8")
    assert "rank_score_100" in csv_text
    assert "collapse_reason" in csv_text
    assert "active_body_radius_norm" in csv_text
    assert "life_topology" in csv_text
    assert "active_axis_circular_span_norm" in csv_text
    assert "continuation_life_gate_pass" in csv_text
    assert "first_failure_horizon" in csv_text
    assert "selection_selected_by" in csv_text
    assert "mutation_precheck_pass" in csv_text
