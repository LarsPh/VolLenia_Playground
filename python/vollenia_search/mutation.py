from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from typing import Any

import torch

from vollenia_diff.params import LeniaParams
from vollenia_search.targets import TargetContext, make_target_context, target_from_context

def _scale_params_radius(params: LeniaParams, scale: float) -> LeniaParams:
    if scale == 1.0:
        return params
    return replace(params, R=float(params.R) * float(scale))

def _normal_scalar(std: float, generator: torch.Generator | None) -> float:
    if float(std) == 0.0:
        return 0.0
    return float(torch.randn((), generator=generator).item()) * float(std)

def _uniform_scalar(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator).item())

def _bernoulli(probability: float, generator: torch.Generator | None) -> bool:
    return _uniform_scalar(generator) < _clamp(float(probability), 0.0, 1.0)

def _choice(values: list[Any], generator: torch.Generator | None) -> Any:
    if not values:
        raise ValueError("Cannot sample from an empty choice list")
    index = int(torch.randint(len(values), (), generator=generator).item())
    return values[index]

def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))

def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]

def _field_probability(config: dict[str, Any], name: str) -> float:
    return float(config.get(f"{name}_probability", 1.0))

def _apply_rule_noise(
    params: LeniaParams,
    config: dict[str, Any] | None,
    *,
    seed: int | None,
    label: str,
    allow_structure: bool = False,
) -> tuple[LeniaParams, dict[str, Any]]:
    config = config or {}
    before = params.to_catalog_dict()
    if not bool(config.get("enabled", False)):
        return params, {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "label": label,
            "before": before,
            "after": before,
            "fields": {},
        }
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    if not _bernoulli(float(config.get("probability", 1.0)), generator):
        return params, {
            "enabled": True,
            "applied": False,
            "reason": "probability",
            "label": label,
            "seed": seed,
            "config": copy.deepcopy(config),
            "before": before,
            "after": before,
            "fields": {},
        }

    R = float(params.R)
    m = float(params.m)
    s = float(params.s)
    T = float(params.T)
    b = [float(value) for value in params.b]
    kn = int(params.kn)
    gn = int(params.gn)
    fields: dict[str, Any] = {}

    if allow_structure and _bernoulli(float(config.get("ring_count_probability", 0.0)), generator):
        choices = [max(int(value), 1) for value in _as_list(config.get("ring_count_choices"))]
        if choices:
            ring_count = int(_choice(choices, generator))
            old_b = b
            b = [old_b[index] if index < len(old_b) else _uniform_scalar(generator) for index in range(ring_count)]
            fields["ring_count"] = {"applied": True, "before": len(old_b), "after": ring_count}
    if allow_structure and _bernoulli(float(config.get("kn_probability", 0.0)), generator):
        choices = [int(value) for value in _as_list(config.get("kn_choices"))]
        if choices:
            old_kn = kn
            kn = int(_choice(choices, generator))
            fields["kn"] = {"applied": True, "before": old_kn, "after": kn}
    if allow_structure and _bernoulli(float(config.get("gn_probability", 0.0)), generator):
        choices = [int(value) for value in _as_list(config.get("gn_choices"))]
        if choices:
            old_gn = gn
            gn = int(_choice(choices, generator))
            fields["gn"] = {"applied": True, "before": old_gn, "after": gn}

    if _bernoulli(_field_probability(config, "R"), generator):
        old = R
        R = R * float(torch.exp(torch.tensor(_normal_scalar(float(config.get("R_log_std", 0.0)), generator))).item())
        fields["R"] = {"applied": R != old, "before": old, "after": R}
    if _bernoulli(_field_probability(config, "m"), generator):
        old = m
        m = _clamp(m + _normal_scalar(float(config.get("m_std", 0.0)), generator), 1.0e-4, 0.499)
        fields["m"] = {"applied": m != old, "before": old, "after": m}
    if _bernoulli(_field_probability(config, "s"), generator):
        old = s
        s = s * float(torch.exp(torch.tensor(_normal_scalar(float(config.get("s_log_std", 0.0)), generator))).item())
        fields["s"] = {"applied": s != old, "before": old, "after": s}
    if _bernoulli(_field_probability(config, "T"), generator):
        old = T
        T = T * float(torch.exp(torch.tensor(_normal_scalar(float(config.get("T_log_std", 0.0)), generator))).item())
        fields["T"] = {"applied": T != old, "before": old, "after": T}
    b_applied = _bernoulli(_field_probability(config, "b"), generator)
    b_std = float(config.get("b_std", 0.0))
    old_b = list(b)
    b_mask: list[bool] = []
    if b_applied:
        element_probability = float(config.get("b_element_probability", 1.0))
        for index, value in enumerate(b):
            element_applied = _bernoulli(element_probability, generator)
            b_mask.append(element_applied)
            if element_applied:
                b[index] = _clamp(float(value) + _normal_scalar(b_std, generator), 0.0, 1.0)
    fields["b"] = {"applied": b != old_b, "before": old_b, "after": b, "mask": b_mask}
    noisy = replace(params, R=max(R, 1.0e-5), m=m, s=max(s, 1.0e-5), T=max(T, 1.0), b=b, kn=kn, gn=gn).sanitized()
    after = noisy.to_catalog_dict()
    changed = after != before
    metadata = {
        "enabled": True,
        "applied": changed,
        "reason": "applied" if changed else "no_field_changed",
        "label": label,
        "seed": seed,
        "config": copy.deepcopy(config),
        "before": before,
        "after": after,
        "fields": fields,
    }
    return noisy, metadata

def _initial_logit_std(mutation: dict[str, Any]) -> float:
    return float(mutation.get("initial_logit_std", 0.0) or 0.0)

def disabled_mutation(mutation: dict[str, Any]) -> dict[str, Any]:
    disabled = copy.deepcopy(mutation)
    disabled["initial_logit_std"] = 0.0
    disabled["rule"] = {**dict(disabled.get("rule", {})), "enabled": False}
    return disabled

def candidate_mutation_decision(
    mutation: dict[str, Any],
    *,
    iteration: int,
    seed: int,
    full_inner_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    full_inner_steps = max(int(full_inner_steps), 1)
    disabled = disabled_mutation(mutation)

    no_mutation_every = int(float(mutation.get("no_mutation_every_iterations") or 0))
    if no_mutation_every > 0 and (int(iteration) + 1) % no_mutation_every == 0:
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "scheduled_no_mutation",
            "iteration": int(iteration),
            "seed": int(seed),
            "inner_steps_used": full_inner_steps,
            "config": copy.deepcopy(mutation),
        }
        return disabled, metadata, full_inner_steps

    if not _bernoulli(float(mutation.get("probability", 1.0)), generator):
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "mutation_probability",
            "iteration": int(iteration),
            "seed": int(seed),
            "inner_steps_used": full_inner_steps,
            "config": copy.deepcopy(mutation),
        }
        return disabled, metadata, full_inner_steps

    effective = copy.deepcopy(mutation)
    initial_std = _initial_logit_std(effective)
    initial_applied = initial_std > 0.0 and _bernoulli(float(effective.get("initial_logit_probability", 1.0)), generator)
    if not initial_applied:
        effective["initial_logit_std"] = 0.0

    rule_config = dict(effective.get("rule", {}))
    rule_applied = bool(rule_config.get("enabled", False)) and _bernoulli(float(rule_config.get("probability", 1.0)), generator)
    if not rule_applied:
        rule_config["enabled"] = False
    effective["rule"] = rule_config

    applied = bool(initial_applied or rule_applied)
    if not applied:
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "no_component_selected",
            "iteration": int(iteration),
            "seed": int(seed),
            "inner_steps_used": full_inner_steps,
            "initial_logit": {"applied": False, "std": 0.0},
            "rule": {"applied": False},
            "config": copy.deepcopy(mutation),
        }
        return disabled, metadata, full_inner_steps

    mutated_inner_steps = effective.get("mutated_inner_optim_steps")
    inner_steps_used = full_inner_steps if mutated_inner_steps is None else min(max(int(float(mutated_inner_steps)), 1), full_inner_steps)
    metadata = {
        "enabled": True,
        "applied": True,
        "reason": "mutation",
        "iteration": int(iteration),
        "seed": int(seed),
        "inner_steps_used": inner_steps_used,
        "initial_logit": {"applied": bool(initial_applied), "std": float(effective.get("initial_logit_std", 0.0) or 0.0)},
        "rule": {"applied": bool(rule_applied)},
        "config": copy.deepcopy(mutation),
    }
    return effective, metadata, inner_steps_used

def apply_candidate_mutation(
    source_state0: torch.Tensor,
    base_params: LeniaParams,
    *,
    mutation: dict[str, Any],
    mutation_seed: int,
) -> tuple[torch.Tensor, LeniaParams, dict[str, Any]]:
    mutated_params, rule_mutation_metadata = _apply_rule_noise(
        base_params,
        dict(mutation.get("rule", {})),
        seed=mutation_seed,
        label="candidate_mutation",
    )
    initial = torch.clamp(source_state0.detach(), 1.0e-4, 1.0 - 1.0e-4)
    logits = torch.logit(initial)
    mutation_std = float(mutation.get("initial_logit_std", 0.0) or 0.0)
    if mutation_std > 0.0:
        generator = torch.Generator(device=logits.device).manual_seed(int(mutation_seed) + 17)
        logits = logits + torch.randn(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator) * mutation_std
    return torch.sigmoid(logits).detach(), mutated_params, rule_mutation_metadata

def precheck_mutation_candidate(
    source_state0: torch.Tensor,
    base_params: LeniaParams,
    *,
    args: argparse.Namespace,
    profile: str,
    target: torch.Tensor | None,
    target_context: TargetContext | None = None,
    mutation: dict[str, Any],
    mutation_decision: dict[str, Any],
    mutation_seed: int,
    progress: SearchProgress,
    search_task: int | None,
) -> tuple[int, dict[str, Any]]:
    precheck = dict(mutation.get("precheck", {}))
    if not bool(precheck.get("enabled", False)) or not bool(mutation_decision.get("applied", False)):
        metadata = {
            "enabled": bool(precheck.get("enabled", False)),
            "attempted": False,
            "pass": True,
            "selected_seed": int(mutation_seed),
            "reason": "disabled_or_no_mutation",
            "attempts": [],
        }
        return int(mutation_seed), metadata

    attempts = max(int(float(precheck.get("max_attempts") or 1)), 1)
    steps = max(int(float(precheck.get("steps") or args.steps)), 1)
    accept_first = bool(precheck.get("accept_first_life_gate_pass", True))
    best_score = float("-inf")
    best_seed = int(mutation_seed)
    best_reason = "not_run"
    best_pass = False
    records: list[dict[str, Any]] = []
    from .evaluate import evaluate_candidate

    for attempt in range(attempts):
        attempt_seed = int(mutation_seed) + attempt * 9973
        candidate_state0, candidate_params, rule_metadata = apply_candidate_mutation(
            source_state0,
            base_params,
            mutation=mutation,
            mutation_seed=attempt_seed,
        )
        attempt_target_context = make_target_context(profile, candidate_state0, args.objective)
        _, metrics, _, score = evaluate_candidate(
            candidate_state0,
            candidate_params,
            profile=profile,
            target=target_from_context(attempt_target_context) if target is None else target,
            target_context=attempt_target_context if target_context is not None else None,
            steps=steps,
            clip_mode=args.eval_clip_mode,
            compile_step=args.compile_step,
            objective=args.objective,
            evaluation={"continuation_steps": [], "stop_on_first_failure": True},
        )
        passed = bool(metrics.get("life_gate_pass", False))
        reason = str(metrics.get("collapse_reason", "ok"))
        record = {
            "attempt": attempt,
            "seed": attempt_seed,
            "score": float(score),
            "score_100": float(metrics.get("score_100", score)),
            "rank_score_100": float(metrics.get("rank_score_100", score)),
            "life_gate_pass": passed,
            "collapse_reason": reason,
            "rule": rule_metadata,
        }
        records.append(record)
        if float(score) > best_score:
            best_score = float(score)
            best_seed = attempt_seed
            best_reason = reason
            best_pass = passed
        progress.update(search_task, status=f"precheck attempt={attempt + 1}/{attempts} gate={passed} reason={reason}")
        if passed and accept_first:
            break
    metadata = {
        "enabled": True,
        "attempted": True,
        "pass": best_pass,
        "steps": steps,
        "selected_seed": best_seed,
        "best_score": best_score,
        "best_reason": best_reason,
        "attempts": records,
    }
    return best_seed, metadata

__all__ = [
    "_apply_rule_noise",
    "apply_candidate_mutation",
    "candidate_mutation_decision",
    "disabled_mutation",
    "precheck_mutation_candidate",
]
