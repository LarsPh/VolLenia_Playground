from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from vollenia_diff.metrics import state_summary
from vollenia_diff.params import LeniaParams
from vollenia_diff.rollout_losses import PROFILE_LOSSES, rollout_collect, terms_to_json
from vollenia_diff.simulator import LeniaSimulator
from vollenia_search.targets import TargetContext, target_from_context

from .mutation import apply_candidate_mutation

def _logit_scalar(value: float, low: float, high: float, device: torch.device) -> torch.Tensor:
    scaled = min(max((float(value) - low) / (high - low), 1.0e-4), 1.0 - 1.0e-4)
    return torch.logit(torch.tensor(scaled, device=device, dtype=torch.float32))

def _inv_softplus(value: float, min_value: float, device: torch.device) -> torch.Tensor:
    shifted = max(float(value) - min_value, 1.0e-4)
    return torch.log(torch.expm1(torch.tensor(shifted, device=device, dtype=torch.float32)))

def _parse_optimize_params(value: str) -> set[str]:
    requested = {part.strip() for part in value.split(",") if part.strip()}
    unsupported = requested - {"m", "s", "T"}
    if unsupported:
        raise ValueError(
            f"Unsupported differentiable params {sorted(unsupported)}; Plan 07 supports only m,s,T. "
            "R/b/kernel differentiability is deferred to a later milestone."
        )
    return requested

def params_from_raw(base: LeniaParams, raw: dict[str, torch.Tensor]) -> tuple[LeniaParams, dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    if "m" in raw:
        tensors["m"] = 0.5 * torch.sigmoid(raw["m"])
    if "s" in raw:
        tensors["s"] = F.softplus(raw["s"]) + 1.0e-4
    if "T" in raw:
        tensors["T"] = F.softplus(raw["T"]) + 1.0
    params = replace(
        base,
        m=float(tensors.get("m", torch.tensor(base.m)).detach().cpu()),
        s=float(tensors.get("s", torch.tensor(base.s)).detach().cpu()),
        T=float(tensors.get("T", torch.tensor(base.T)).detach().cpu()),
    )
    return params, tensors

def make_raw_params(base: LeniaParams, requested: set[str], device: torch.device) -> dict[str, torch.Tensor]:
    raw: dict[str, torch.Tensor] = {}
    if "m" in requested:
        raw["m"] = _logit_scalar(base.m, 0.0, 0.5, device).detach().requires_grad_(True)
    if "s" in requested:
        raw["s"] = _inv_softplus(base.s, 1.0e-4, device).detach().requires_grad_(True)
    if "T" in requested:
        raw["T"] = _inv_softplus(base.T, 1.0, device).detach().requires_grad_(True)
    return raw

def make_optimizer(args: argparse.Namespace, logits: torch.Tensor, raw_params: dict[str, torch.Tensor]) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, str(args.optimizer_name), None)
    if optimizer_class is None:
        raise ValueError(f"Unknown torch optimizer: {args.optimizer_name}")
    base_lr = float(args.optimizer_lr)
    initial_lr = args.optimizer_initial_lr
    param_lr = args.optimizer_param_lr
    if initial_lr is None and param_lr is None:
        return optimizer_class([logits, *raw_params.values()], lr=base_lr)
    groups: list[dict[str, Any]] = [{"params": [logits], "lr": float(initial_lr if initial_lr is not None else base_lr)}]
    if raw_params:
        groups.append({"params": list(raw_params.values()), "lr": float(param_lr if param_lr is not None else base_lr)})
    return optimizer_class(groups, lr=base_lr)

def _tensor_grad_norm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        return torch.zeros((), device=tensor.device)
    return tensor.grad.detach().norm()

def gradient_diagnostics(
    logits: torch.Tensor,
    raw_params: dict[str, torch.Tensor],
    *,
    max_grad_norm: float | None,
) -> dict[str, float]:
    params = [logits, *raw_params.values()]
    global_before = torch.sqrt(
        sum((_tensor_grad_norm(param) ** 2 for param in params if param.grad is not None), torch.zeros((), device=logits.device))
    )
    clipped_to: float | None = None
    if max_grad_norm is not None and max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        clipped_to = float(max_grad_norm)
    global_after = torch.sqrt(
        sum((_tensor_grad_norm(param) ** 2 for param in params if param.grad is not None), torch.zeros((), device=logits.device))
    )
    logit_norm = _tensor_grad_norm(logits)
    diagnostics = {
        "grad_norm_global": float(global_before.detach().cpu()),
        "grad_norm_global_after_clip": float(global_after.detach().cpu()),
        "logit_grad_norm": float(logit_norm.detach().cpu()),
        "logit_grad_rms": float((logit_norm / (logits.numel() ** 0.5)).detach().cpu()),
        "max_grad_norm": float(clipped_to) if clipped_to is not None else 0.0,
    }
    for name, tensor in raw_params.items():
        diagnostics[f"{name}_grad"] = float(_tensor_grad_norm(tensor).detach().cpu())
    for name in ("m", "s", "T"):
        diagnostics.setdefault(f"{name}_grad", 0.0)
    return diagnostics

def optimize_candidate(
    source_state0: torch.Tensor,
    base_params: LeniaParams,
    *,
    args: argparse.Namespace,
    profile: str,
    target: torch.Tensor | None,
    target_context: TargetContext | None = None,
    steps: int,
    inner_steps: int,
    train_clip_mode: str,
    optimize_params: set[str],
    mutation_std: float,
    mutation: dict[str, Any],
    mutation_decision: dict[str, Any],
    mutation_seed: int,
    compile_step: bool,
    progress: SearchProgress,
    progress_task: int | None,
) -> tuple[torch.Tensor, LeniaParams, dict[str, Any], dict[str, float], list[dict[str, float]], dict[str, float]]:
    mutated_state0, base_params, rule_mutation_metadata = apply_candidate_mutation(
        source_state0,
        base_params,
        mutation=mutation,
        mutation_seed=mutation_seed,
    )
    initial = torch.clamp(mutated_state0.detach(), 1.0e-4, 1.0 - 1.0e-4)
    logits = torch.logit(initial).detach().requires_grad_(True)
    raw_params = make_raw_params(base_params, optimize_params, source_state0.device)
    optimizer = make_optimizer(args, logits, raw_params)
    simulator = LeniaSimulator(source_state0.shape, base_params, device=source_state0.device, clip_mode=train_clip_mode, compile_step=compile_step)
    grad_history: list[dict[str, float]] = []
    last_grad_diagnostics: dict[str, float] = {
        "grad_norm_global": 0.0,
        "grad_norm_global_after_clip": 0.0,
        "logit_grad_norm": 0.0,
        "logit_grad_rms": 0.0,
        "m_grad": 0.0,
        "s_grad": 0.0,
        "T_grad": 0.0,
        "max_grad_norm": 0.0,
    }
    last_terms: dict[str, float] = {}
    final_train = source_state0
    params = base_params
    if target_context is not None:
        target = target_from_context(target_context)
    inner_log_every = max(int(float(getattr(args, "logging", {}).get("inner_log_every", 5))), 1)
    detailed_terms_every = max(int(float(getattr(args, "logging", {}).get("detailed_terms_every", 25))), 1)

    for step_index in range(int(inner_steps)):
        optimizer.zero_grad(set_to_none=True)
        state0 = torch.sigmoid(logits)
        params, tensor_params = params_from_raw(base_params, raw_params)
        states, state_steps = rollout_collect(
            simulator,
            state0,
            steps,
            params=base_params,
            m=tensor_params.get("m"),
            s=tensor_params.get("s"),
            T=tensor_params.get("T"),
            clip_mode=train_clip_mode,
            sample_interval=max(steps // 4, 1),
            return_steps=True,
        )
        loss_fn = PROFILE_LOSSES[profile]
        if target is None:
            loss, terms = loss_fn(states, objective=args.objective)
        else:
            loss, terms = loss_fn(states, target=target, objective=args.objective, state_steps=state_steps, total_steps=steps, target_context=target_context)
        loss.backward()
        last_grad_diagnostics = gradient_diagnostics(logits, raw_params, max_grad_norm=args.max_grad_norm)
        grad_history.append(last_grad_diagnostics)
        optimizer.step()
        final_train = states[-1].detach()
        should_log_terms = (
            step_index == int(inner_steps) - 1
            or (step_index + 1) % inner_log_every == 0
            or (step_index + 1) % detailed_terms_every == 0
        )
        if should_log_terms:
            last_terms = terms_to_json(terms)
        else:
            last_terms = {"loss_total": float(loss.detach().cpu())}
        term_status = " ".join(
            f"{key}={last_terms[key]:.3g}"
            for key in ("com", "target_mask", "mass_ratio", "active_ratio", "compactness_ratio", "visibility")
            if key in last_terms
        )
        progress.update(
            progress_task,
            advance=1,
            status=(
                f"step={step_index + 1} loss={last_terms.get('loss_total', 0.0):.4g} "
                f"logit_rms={last_grad_diagnostics['logit_grad_rms']:.3g} "
                f"m={last_grad_diagnostics['m_grad']:.3g} s={last_grad_diagnostics['s_grad']:.3g} "
                f"T={last_grad_diagnostics['T_grad']:.3g} {term_status}"
            ),
        )

    learned_state0 = torch.sigmoid(logits).detach()
    params, _ = params_from_raw(base_params, raw_params)
    final_summary = state_summary(final_train.detach(), target)
    if getattr(simulator, "dynamic_compile_error", None):
        final_summary["compile_metadata"] = {"dynamic_step_fallback": True, "error": simulator.dynamic_compile_error}
    if target_context is not None:
        final_summary["target_context"] = target_context.to_json()
    final_summary["candidate_mutation"] = {
        **copy.deepcopy(mutation_decision),
        "initial_logit_std": float(mutation_std),
        "rule": rule_mutation_metadata,
    }
    return learned_state0, params, final_summary, last_terms, grad_history, last_grad_diagnostics

__all__ = [
    "_parse_optimize_params",
    "gradient_diagnostics",
    "make_optimizer",
    "make_raw_params",
    "optimize_candidate",
    "params_from_raw",
]
