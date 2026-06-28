from __future__ import annotations

import copy
from typing import Any

import torch

from vollenia_diff.metrics import descriptors_to_json, normalized_descriptors, state_summary
from vollenia_diff.params import LeniaParams
from vollenia_diff.rollout_losses import PROFILE_LOSSES, rollout_collect, target_from_initial_offset, terms_to_json
from vollenia_diff.simulator import LeniaSimulator
from vollenia_search.targets import TargetContext, make_target_context, target_from_context

from .archive import score_100_from_loss

def target_for_profile(profile: str, initial: torch.Tensor, objective: dict[str, Any] | None = None) -> torch.Tensor | None:
    if profile != "move_shape_target":
        return None
    objective = objective or {}
    offset_norm = objective.get("target_offset_norm_zyx", [0.0, 0.0, 0.12])
    return target_from_initial_offset(initial, list(offset_norm))

def target_context_for_profile(profile: str, initial: torch.Tensor, objective: dict[str, Any] | None = None) -> TargetContext:
    return make_target_context(profile, initial, objective)

def _range_violation(value: float, low: float, high: float, low_reason: str, high_reason: str) -> str | None:
    if value < low:
        return low_reason
    if value > high:
        return high_reason
    return None

def _float_list(value: Any, *, length: int, default: list[float]) -> list[float]:
    if isinstance(value, list | tuple):
        result = [float(item) for item in value]
    else:
        result = list(default)
    if len(result) < length:
        result.extend(default[len(result) : length])
    return result[:length]

def _topology_from_full_axis_count(full_axis_count: int) -> str:
    if full_axis_count <= 0:
        return "blob"
    if full_axis_count == 1:
        return "cylinder"
    if full_axis_count == 2:
        return "plane"
    return "global_noise"

def evaluate_life_gate(descriptor: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    gates = objective.get("eval_gates", {}) if isinstance(objective, dict) else {}
    if not isinstance(gates, dict) or not gates:
        return {"life_gate_pass": True, "collapse_reason": "ok", "gate_violations": [], "eval_gates": {}}

    violations: list[dict[str, Any]] = []

    def add(reason: str, name: str, value: float, threshold: Any) -> None:
        violations.append({"reason": reason, "metric": name, "value": value, "threshold": threshold})

    topology_config = gates.get("topology")
    topology_enabled = isinstance(topology_config, dict) and bool(topology_config.get("enabled", False))
    max_density = float(descriptor.get("max_density", 0.0))
    min_max_density = gates.get("min_max_density")
    if min_max_density is not None and max_density < float(min_max_density):
        add("dead_density", "max_density", max_density, float(min_max_density))

    if topology_enabled:
        active_axis_span = _float_list(
            descriptor.get("active_axis_circular_span_norm"),
            length=3,
            default=[0.0, 0.0, 0.0],
        )
        active_axis_coverage = _float_list(
            descriptor.get("active_axis_coverage_norm"),
            length=3,
            default=active_axis_span,
        )
        axis_border = _float_list(descriptor.get("axis_border_mass"), length=3, default=[0.0, 0.0, 0.0])
        full_axis_span_min = float(topology_config.get("full_axis_span_min", 0.85))
        full_axes = [axis for axis, span in enumerate(active_axis_span) if float(span) >= full_axis_span_min]
        compact_axes = [axis for axis in range(3) if axis not in full_axes]
        full_axis_count = len(full_axes)
        life_topology = _topology_from_full_axis_count(full_axis_count)
        allowed = [str(value) for value in topology_config.get("allowed", ["blob", "cylinder", "plane"])]

        by_type = topology_config.get("by_type", {})
        type_gates = dict(by_type.get(life_topology, {})) if isinstance(by_type, dict) and isinstance(by_type.get(life_topology, {}), dict) else {}

        mass_fraction = float(descriptor.get("mass_fraction", 0.0))
        mass_window = type_gates.get("mass_fraction", gates.get("mass_fraction"))
        if mass_window is not None:
            low, high = [float(value) for value in mass_window]
            reason = _range_violation(mass_fraction, low, high, "dead_mass", "over_mass_noise")
            if reason:
                add(reason, "mass_fraction", mass_fraction, [low, high])

        active_fraction = float(descriptor.get("active_fraction", 0.0))
        active_window = type_gates.get("active_fraction", gates.get("active_fraction"))
        if active_window is not None:
            low, high = [float(value) for value in active_window]
            reason = _range_violation(active_fraction, low, high, "under_active", "over_active_noise")
            if reason:
                add(reason, "active_fraction", active_fraction, [low, high])

        if life_topology == "global_noise":
            add("all_axes_full_noise", "full_axis_count", float(full_axis_count), {"max_allowed": 2})
        elif life_topology not in allowed:
            add("topology_filtered", "life_topology", float(full_axis_count), allowed)

        if life_topology == "blob":
            body_radius_norm = float(descriptor.get("body_radius_norm", 0.0))
            body_radius_window = type_gates.get("body_radius_norm", gates.get("body_radius_norm"))
            if body_radius_window is not None:
                low, high = [float(value) for value in body_radius_window]
                reason = _range_violation(body_radius_norm, low, high, "too_compact", "too_diffuse")
                if reason:
                    add(reason, "body_radius_norm", body_radius_norm, [low, high])

            active_body_radius_norm = float(descriptor.get("active_body_radius_norm", 0.0))
            active_body_radius_window = type_gates.get("active_body_radius_norm", gates.get("active_body_radius_norm"))
            if active_body_radius_window is not None:
                low, high = [float(value) for value in active_body_radius_window]
                reason = _range_violation(active_body_radius_norm, low, high, "active_too_compact", "active_too_diffuse_noise")
                if reason:
                    add(reason, "active_body_radius_norm", active_body_radius_norm, [low, high])

            border_mass = float(descriptor.get("border_mass", 0.0))
            max_border_mass = type_gates.get("max_border_mass", gates.get("max_border_mass"))
            if max_border_mass is not None and border_mass > float(max_border_mass):
                add("border_leak", "border_mass", border_mass, float(max_border_mass))
        elif life_topology in {"cylinder", "plane"}:
            compact_window = type_gates.get("compact_axis_span_norm", topology_config.get("compact_axis_span_norm"))
            if compact_window is not None:
                low, high = [float(value) for value in compact_window]
            else:
                low = 0.0
                high = float(topology_config.get("compact_axis_span_max", 0.72))
            for axis in compact_axes:
                span = float(active_axis_span[axis])
                reason = _range_violation(span, low, high, "compact_axis_too_empty", "compact_axis_too_wide_noise")
                if reason:
                    add(reason, f"active_axis_circular_span_norm[{axis}]", span, [low, high])
            compact_axis_border_max = type_gates.get(
                "compact_axis_border_mass_max",
                topology_config.get("compact_axis_border_mass_max"),
            )
            if compact_axis_border_max is not None:
                for axis in compact_axes:
                    value = float(axis_border[axis])
                    if value > float(compact_axis_border_max):
                        add("compact_axis_border_leak", f"axis_border_mass[{axis}]", value, float(compact_axis_border_max))

        reason_priority = [
            "dead_density",
            "dead_mass",
            "under_active",
            "all_axes_full_noise",
            "topology_filtered",
            "compact_axis_too_wide_noise",
            "over_active_noise",
            "over_mass_noise",
            "compact_axis_border_leak",
            "active_too_diffuse_noise",
            "too_diffuse",
            "border_leak",
            "compact_axis_too_empty",
            "active_too_compact",
            "too_compact",
        ]
        reasons = [item["reason"] for item in violations]
        collapse_reason = "ok"
        for reason in reason_priority:
            if reason in reasons:
                collapse_reason = reason
                break
        return {
            "life_gate_pass": not violations,
            "collapse_reason": collapse_reason,
            "gate_violations": violations,
            "eval_gates": copy.deepcopy(gates),
            "life_topology": life_topology,
            "full_axis_count": full_axis_count,
            "full_axes": full_axes,
            "compact_axes": compact_axes,
            "active_axis_coverage_norm": active_axis_coverage,
            "active_axis_circular_span_norm": active_axis_span,
            "axis_border_mass": axis_border,
        }

    mass_fraction = float(descriptor.get("mass_fraction", 0.0))
    mass_window = gates.get("mass_fraction")
    if mass_window is not None:
        low, high = [float(value) for value in mass_window]
        reason = _range_violation(mass_fraction, low, high, "dead_mass", "over_mass_noise")
        if reason:
            add(reason, "mass_fraction", mass_fraction, [low, high])

    active_fraction = float(descriptor.get("active_fraction", 0.0))
    active_window = gates.get("active_fraction")
    if active_window is not None:
        low, high = [float(value) for value in active_window]
        reason = _range_violation(active_fraction, low, high, "under_active", "over_active_noise")
        if reason:
            add(reason, "active_fraction", active_fraction, [low, high])

    body_radius_norm = float(descriptor.get("body_radius_norm", 0.0))
    body_radius_window = gates.get("body_radius_norm")
    if body_radius_window is not None:
        low, high = [float(value) for value in body_radius_window]
        reason = _range_violation(body_radius_norm, low, high, "too_compact", "too_diffuse")
        if reason:
            add(reason, "body_radius_norm", body_radius_norm, [low, high])

    active_body_radius_norm = float(descriptor.get("active_body_radius_norm", 0.0))
    active_body_radius_window = gates.get("active_body_radius_norm")
    if active_body_radius_window is not None:
        low, high = [float(value) for value in active_body_radius_window]
        reason = _range_violation(active_body_radius_norm, low, high, "active_too_compact", "active_too_diffuse_noise")
        if reason:
            add(reason, "active_body_radius_norm", active_body_radius_norm, [low, high])

    border_mass = float(descriptor.get("border_mass", 0.0))
    max_border_mass = gates.get("max_border_mass")
    if max_border_mass is not None and border_mass > float(max_border_mass):
        add("border_leak", "border_mass", border_mass, float(max_border_mass))

    reason_priority = [
        "dead_density",
        "dead_mass",
        "under_active",
        "over_active_noise",
        "over_mass_noise",
        "active_too_diffuse_noise",
        "too_diffuse",
        "border_leak",
        "active_too_compact",
        "too_compact",
    ]
    reasons = [item["reason"] for item in violations]
    collapse_reason = "ok"
    for reason in reason_priority:
        if reason in reasons:
            collapse_reason = reason
            break
    return {
        "life_gate_pass": not violations,
        "collapse_reason": collapse_reason,
        "gate_violations": violations,
        "eval_gates": copy.deepcopy(gates),
    }

def _continuation_horizons(evaluation: dict[str, Any] | None, primary_steps: int) -> list[int]:
    if not isinstance(evaluation, dict):
        return []
    values = evaluation.get("continuation_steps", [])
    if not isinstance(values, list | tuple):
        return []
    horizons = sorted({int(value) for value in values if int(value) > int(primary_steps)})
    return horizons

def evaluate_continuation_gates(
    state0: torch.Tensor,
    primary_final: torch.Tensor,
    params: LeniaParams,
    *,
    target: torch.Tensor | None,
    primary_steps: int,
    clip_mode: str,
    compile_step: bool,
    objective: dict[str, Any],
    evaluation: dict[str, Any] | None,
    progress: SearchProgress | None = None,
    progress_task: int | None = None,
) -> dict[str, Any]:
    horizons = _continuation_horizons(evaluation, primary_steps)
    stop_on_first_failure = bool((evaluation or {}).get("stop_on_first_failure", False)) if isinstance(evaluation, dict) else False
    if not horizons:
        return {
            "continuation_steps": [],
            "continuation_results": [],
            "continuation_life_gate_pass": True,
            "continuation_collapse_reason": "ok",
            "continuation_gate_violations": [],
            "first_failure_horizon": None,
            "longest_passed_horizon": int(primary_steps),
            "failed_at_final_horizon": False,
            "skipped_continuation_steps": [],
        }

    simulator = LeniaSimulator(state0.shape, params, device=state0.device, clip_mode=clip_mode, compile_step=compile_step)
    state = primary_final.detach()
    current_step = int(primary_steps)
    results: list[dict[str, Any]] = []
    gate_violations: list[dict[str, Any]] = []
    continuation_pass = True
    continuation_reason = "ok"
    first_failure_horizon: int | None = None
    longest_passed_horizon = int(primary_steps)
    skipped_steps: list[int] = []

    with torch.no_grad():
        for horizon_index, horizon in enumerate(horizons):
            for _ in range(horizon - current_step):
                state = simulator.step(state)
            current_step = horizon
            descriptor = descriptors_to_json(normalized_descriptors(state, initial=state0, target=target))
            summary = state_summary(state, target)
            gate = evaluate_life_gate({**summary, **descriptor}, objective)
            result = {
                "steps": horizon,
                **summary,
                "descriptor": descriptor,
                **gate,
            }
            results.append(result)
            if not gate["life_gate_pass"]:
                continuation_pass = False
                first_failure_horizon = int(horizon) if first_failure_horizon is None else first_failure_horizon
                if continuation_reason == "ok":
                    continuation_reason = str(gate["collapse_reason"])
                for violation in gate["gate_violations"]:
                    gate_violations.append({"steps": horizon, **violation})
            else:
                longest_passed_horizon = int(horizon)
            if progress is not None:
                progress.update(
                    progress_task,
                    advance=1,
                    status=f"continuation={horizon} gate={gate['life_gate_pass']} reason={gate['collapse_reason']}",
                )
            if stop_on_first_failure and not gate["life_gate_pass"]:
                skipped_steps = horizons[horizon_index + 1 :]
                if progress is not None and skipped_steps:
                    progress.update(progress_task, advance=len(skipped_steps), status=f"skipped_after={horizon} reason={gate['collapse_reason']}")
                break

    return {
        "continuation_steps": horizons,
        "continuation_results": results,
        "continuation_life_gate_pass": continuation_pass,
        "continuation_collapse_reason": continuation_reason,
        "continuation_gate_violations": gate_violations,
        "first_failure_horizon": first_failure_horizon,
        "longest_passed_horizon": longest_passed_horizon,
        "failed_at_final_horizon": first_failure_horizon == horizons[-1] if first_failure_horizon is not None else False,
        "skipped_continuation_steps": skipped_steps,
    }

def evaluate_candidate(
    state0: torch.Tensor,
    params: LeniaParams,
    *,
    profile: str,
    target: torch.Tensor | None,
    target_context: TargetContext | None = None,
    steps: int,
    clip_mode: str,
    compile_step: bool,
    objective: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
    progress: SearchProgress | None = None,
    progress_task: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, float], float]:
    if target_context is not None:
        target = target_from_context(target_context)
    simulator = LeniaSimulator(state0.shape, params, device=state0.device, clip_mode=clip_mode, compile_step=compile_step)
    states, state_steps = rollout_collect(
        simulator,
        state0,
        steps,
        params=params,
        clip_mode=clip_mode,
        sample_interval=max(steps // 4, 1),
        return_steps=True,
    )
    loss_fn = PROFILE_LOSSES[profile]
    if target is None:
        loss, terms = loss_fn(states, objective=objective)
    else:
        loss, terms = loss_fn(states, target=target, objective=objective, state_steps=state_steps, total_steps=steps, target_context=target_context)
    final = states[-1]
    descriptor_tensor = normalized_descriptors(final, initial=state0, target=target)
    descriptor = descriptors_to_json(descriptor_tensor)
    summary = state_summary(final, target)
    loss_value = float(loss.detach().cpu())
    loss_score = -loss_value
    score_100 = score_100_from_loss(loss_value)
    gate = evaluate_life_gate({**summary, **descriptor}, objective)
    if progress is not None:
        progress.update(progress_task, advance=1, status=f"primary={steps} gate={gate['life_gate_pass']} reason={gate['collapse_reason']}")
    continuation = evaluate_continuation_gates(
        state0,
        final,
        params,
        target=target,
        primary_steps=steps,
        clip_mode=clip_mode,
        compile_step=compile_step,
        objective=objective,
        evaluation=evaluation,
        progress=progress,
        progress_task=progress_task,
    )
    primary_life_gate_pass = bool(gate["life_gate_pass"])
    continuation_life_gate_pass = bool(continuation["continuation_life_gate_pass"])
    life_gate_pass = primary_life_gate_pass and continuation_life_gate_pass
    if primary_life_gate_pass:
        collapse_reason = "ok" if continuation_life_gate_pass else f"continuation_{continuation['continuation_collapse_reason']}"
    else:
        collapse_reason = str(gate["collapse_reason"])
    gate_violations = list(gate["gate_violations"]) + list(continuation["continuation_gate_violations"])
    gate_fail_score_cap = float(objective.get("gate_fail_score_cap", 5.0)) if isinstance(objective, dict) else 5.0
    rank_score_100 = score_100 if life_gate_pass else min(score_100, gate_fail_score_cap)
    score = rank_score_100
    terms_json = terms_to_json(terms)
    terms_json["loss_score"] = loss_score
    terms_json["score_100"] = score_100
    terms_json["rank_score_100"] = rank_score_100
    terms_json["continuation_life_gate_pass"] = continuation_life_gate_pass
    metrics = {
        **summary,
        "descriptor": descriptor,
        **gate,
        **continuation,
        "primary_life_gate_pass": primary_life_gate_pass,
        "primary_collapse_reason": gate["collapse_reason"],
        "primary_gate_violations": gate["gate_violations"],
        "life_gate_pass": life_gate_pass,
        "collapse_reason": collapse_reason,
        "gate_violations": gate_violations,
        "score_100": score_100,
        "rank_score_100": rank_score_100,
        "loss_score": loss_score,
    }
    if getattr(simulator, "dynamic_compile_error", None):
        metrics["compile_metadata"] = {"dynamic_step_fallback": True, "error": simulator.dynamic_compile_error}
    if target_context is not None:
        metrics["target_context"] = target_context.to_json()
    return final, metrics, terms_json, score

__all__ = [
    "_continuation_horizons",
    "evaluate_candidate",
    "evaluate_continuation_gates",
    "evaluate_life_gate",
    "target_context_for_profile",
    "target_for_profile",
]
