from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from . import metrics
from .params import LeniaParams
from .simulator import LeniaSimulator, _clip_mode_id, lenia_step_tensor


@dataclass(slots=True)
class GoalProfile:
    name: str
    weights: dict[str, float] = field(default_factory=dict)


MOVE_SHAPE_TARGET_OBJECTIVE: dict[str, object] = {
    "weights": {
        "com": 8.0,
        "balanced_target": 4.0,
        "target_mask": 0.0,
        "absolute_occupancy": 35.0,
        "mass_ratio": 8.0,
        "active_ratio": 6.0,
        "compactness_ratio": 5.0,
        "anisotropy": 0.2,
        "border": 6.0,
        "visibility": 20.0,
    },
    "windows": {
        "mass_ratio": [0.35, 1.6],
        "active_ratio": [0.25, 1.5],
        "compactness_ratio": [0.25, 1.8],
        "mass_fraction": [0.0001, 0.06],
        "active_fraction": [0.0005, 0.12],
    },
    "target_radius_norm": 0.08,
    "target_density": 0.8,
    "target_falloff_floor": 0.25,
    "anisotropy_max": 8.0,
    "min_active_fraction_soft": 0.01,
    "min_max_density": 0.08,
}


RESCUE_UNSTABLE_ANIMAL_OBJECTIVE: dict[str, object] = {
    "weights": {
        "mass_ratio": 25.0,
        "active_ratio": 10.0,
        "absolute_occupancy": 8.0,
        "compactness_ratio": 5.0,
        "anisotropy": 0.25,
        "border": 2.0,
        "visibility": 8.0,
    },
    "windows": {
        "mass_ratio": [0.4, 2.2],
        "active_ratio": [0.3, 2.5],
        "mass_fraction": [0.00005, 0.18],
        "active_fraction": [0.0002, 0.35],
        "compactness_ratio": [0.35, 3.0],
    },
    "anisotropy_max": 10.0,
    "min_active_fraction_soft": 0.01,
    "min_max_density": 0.08,
}


def _deep_update(base: dict[str, object], updates: dict[str, object] | None) -> dict[str, object]:
    result = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _weights(config: dict[str, object], overrides: dict[str, float] | None) -> dict[str, float]:
    result = {key: float(value) for key, value in dict(config.get("weights", {})).items()}
    result.update({key: float(value) for key, value in (overrides or {}).items()})
    return result


def _window(config: dict[str, object], name: str) -> tuple[float, float]:
    values = dict(config.get("windows", {}))[name]
    low, high = values  # type: ignore[misc]
    return float(low), float(high)


def rollout_collect(
    simulator: LeniaSimulator,
    state0: torch.Tensor,
    steps: int,
    *,
    params: LeniaParams | None = None,
    m: float | torch.Tensor | None = None,
    s: float | torch.Tensor | None = None,
    T: float | torch.Tensor | None = None,
    clip_mode: str | None = None,
    sample_interval: int = 1,
    return_steps: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[int]]:
    base_params = (params or simulator.params).sanitized()
    clip_id = _clip_mode_id(clip_mode or simulator.clip_mode)
    use_simulator_step = m is None and s is None and T is None and (clip_mode is None or clip_mode == simulator.clip_mode)
    state = state0.to(device=simulator.device, dtype=simulator.dtype)
    states = [state]
    state_steps = [0]
    for step_index in range(1, int(steps) + 1):
        if use_simulator_step:
            state = simulator.step(state)
        elif simulator.compile_step:
            state = simulator.dynamic_step(
                state,
                m=base_params.m if m is None else m,
                s=base_params.s if s is None else s,
                T=base_params.T if T is None else T,
            )
        else:
            state = lenia_step_tensor(
                state,
                simulator.kernel_hat,
                simulator.shape,
                base_params.m if m is None else m,
                base_params.s if s is None else s,
                base_params.T if T is None else T,
                base_params.gn,
                clip_id,
            )
        should_sample = sample_interval > 0 and step_index % sample_interval == 0
        if should_sample or step_index == int(steps):
            states.append(state)
            state_steps.append(step_index)
    if return_steps:
        return states, state_steps
    return states


def ratio_window_loss(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    low_t = torch.as_tensor(low, device=value.device, dtype=value.dtype)
    high_t = torch.as_tensor(high, device=value.device, dtype=value.dtype)
    return F.relu(low_t - value) ** 2 + F.relu(value - high_t) ** 2


def fraction_window_loss(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return ratio_window_loss(value, low, high)


def target_sphere_mask(shape: tuple[int, ...], target: torch.Tensor, radius: float, falloff_floor: float = 0.25) -> torch.Tensor:
    device = target.device
    dtype = target.dtype
    axes = [torch.arange(size, device=device, dtype=dtype) for size in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(grids, dim=0)
    centered = coords - target.reshape((-1,) + (1,) * len(shape))
    distance = torch.sqrt((centered * centered).sum(dim=0))
    core = (distance <= radius).to(dtype)
    falloff = torch.clamp(1.0 - (distance - radius) / max(radius, 1.0), 0.0, 1.0)
    return torch.maximum(core, float(falloff_floor) * falloff)


def balanced_target_loss(state: torch.Tensor, target_mask: torch.Tensor, target_density: float) -> torch.Tensor:
    target = target_mask * float(target_density)
    foreground = target_mask > 0.0
    if foreground.any():
        foreground_loss = F.mse_loss(state[foreground], target[foreground])
    else:
        foreground_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    background = ~foreground
    if background.any():
        background_loss = (state[background] * state[background]).mean()
    else:
        background_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    return foreground_loss + background_loss


def target_from_initial_offset(initial: torch.Tensor, offset_norm_zyx: list[float] | tuple[float, ...]) -> torch.Tensor:
    com = metrics.normalized_descriptors(initial)["com_norm"] * torch.tensor(
        [max(size - 1, 1) for size in initial.shape],
        device=initial.device,
        dtype=initial.dtype,
    )
    offset = torch.tensor([float(value) * float(min(initial.shape)) for value in offset_norm_zyx], device=initial.device, dtype=initial.dtype)
    max_coord = torch.tensor([size - 1 for size in initial.shape], device=initial.device, dtype=initial.dtype)
    return torch.minimum(torch.maximum(com + offset, torch.zeros_like(com)), max_coord)


def _base_terms(final: torch.Tensor, initial: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    desc = metrics.normalized_descriptors(final, initial=initial, target=target)
    soft_fraction = metrics.soft_active_fraction(final)
    soft_ratio = metrics.soft_active_ratio(final, initial)
    return {
        "mass_ratio": desc["mass_ratio"],
        "active_ratio": desc["active_ratio"],
        "active_ratio_soft": soft_ratio,
        "compactness_ratio": desc["compactness_ratio"],
        "mass_fraction": desc["mass_fraction"],
        "active_fraction": desc["active_fraction"],
        "active_fraction_soft": soft_fraction,
        "body_radius_norm": desc["body_radius_norm"],
        "anisotropy": desc["anisotropy"],
        "border_mass": desc["border_mass"],
        "max_density": metrics.max_density(final),
    }


def _nearest_state_for_step(states: list[torch.Tensor], state_steps: list[int] | None, desired_step: int) -> tuple[torch.Tensor, int]:
    if not states:
        raise ValueError("states must contain at least one tensor")
    if not state_steps or len(state_steps) != len(states):
        return states[-1], len(states) - 1
    index = min(range(len(state_steps)), key=lambda item: abs(int(state_steps[item]) - int(desired_step)))
    return states[index], int(state_steps[index])


def _target_losses_for_state(
    state: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: float,
    target_density: float,
    falloff_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    min_dim = float(min(state.shape))
    target_norm = metrics.target_distance(state, target) / min_dim
    mask = target_sphere_mask(
        tuple(int(v) for v in state.shape),
        target,
        radius=radius,
        falloff_floor=falloff_floor,
    )
    target_mask_loss = F.mse_loss(state, mask * float(target_density))
    balanced_loss = balanced_target_loss(state, mask, float(target_density))
    return target_norm, target_norm * target_norm, balanced_loss, target_mask_loss


def _context_stage_target(target_context: object | None, index: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if target_context is None:
        return None
    stages = getattr(target_context, "stages", None)
    if not stages or index >= len(stages):
        return None
    values = getattr(stages[index], "target_zyx", None)
    if not values:
        return None
    return torch.tensor([float(value) for value in values], device=device, dtype=dtype)


def move_shape_target_loss(
    states: list[torch.Tensor],
    *,
    target: torch.Tensor,
    weights: dict[str, float] | None = None,
    objective: dict[str, object] | None = None,
    state_steps: list[int] | None = None,
    total_steps: int | None = None,
    target_context: object | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = _deep_update(MOVE_SHAPE_TARGET_OBJECTIVE, objective)
    weights = _weights(config, weights)
    initial = states[0]
    final = states[-1]
    min_dim = float(min(final.shape))
    terms = _base_terms(final, initial, target)
    radius = max(min_dim * float(config["target_radius_norm"]), 2.0)
    target_profile = config.get("target_profile")
    target_profile_enabled = isinstance(target_profile, dict) and target_profile.get("mode") == "staged_offsets"
    stage_terms: dict[str, torch.Tensor] = {}
    if target_profile_enabled:
        target_norm = torch.zeros((), device=final.device, dtype=final.dtype)
        com_loss = torch.zeros((), device=final.device, dtype=final.dtype)
        balanced_loss = torch.zeros((), device=final.device, dtype=final.dtype)
        target_mask_loss = torch.zeros((), device=final.device, dtype=final.dtype)
        stages = list(target_profile.get("stages", []))
        final_step = int(total_steps if total_steps is not None else (state_steps[-1] if state_steps else len(states) - 1))
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            if "at_step" in stage:
                desired_step = int(float(stage["at_step"]))
            else:
                desired_step = int(round(float(stage.get("at_step_fraction", 1.0)) * float(final_step)))
            stage_state, sampled_step = _nearest_state_for_step(states, state_steps, desired_step)
            offset = stage.get("offset_norm_zyx", config.get("target_offset_norm_zyx", [0.0, 0.0, 0.12]))
            stage_target = _context_stage_target(target_context, index, stage_state.device, stage_state.dtype)
            if stage_target is None:
                stage_target = target_from_initial_offset(initial, list(offset))
            stage_target_norm, stage_com_loss, stage_balanced_loss, stage_target_mask_loss = _target_losses_for_state(
                stage_state,
                stage_target,
                radius=radius,
                target_density=float(config["target_density"]),
                falloff_floor=float(config["target_falloff_floor"]),
            )
            scales = dict(stage.get("weight_scale", {}))
            com_scale = float(scales.get("com", 1.0))
            balanced_scale = float(scales.get("balanced_target", 1.0))
            target_mask_scale = float(scales.get("target_mask", 1.0))
            target_norm = stage_target_norm
            com_loss = com_loss + stage_com_loss * com_scale
            balanced_loss = balanced_loss + stage_balanced_loss * balanced_scale
            target_mask_loss = target_mask_loss + stage_target_mask_loss * target_mask_scale
            stage_terms[f"stage_{index}_sampled_step"] = torch.as_tensor(float(sampled_step), device=final.device, dtype=final.dtype)
            stage_terms[f"stage_{index}_target_distance_norm"] = stage_target_norm
            stage_terms[f"stage_{index}_com"] = stage_com_loss
            stage_terms[f"stage_{index}_balanced_target"] = stage_balanced_loss
            stage_terms[f"stage_{index}_target_mask"] = stage_target_mask_loss
        if not stages:
            target_norm, com_loss, balanced_loss, target_mask_loss = _target_losses_for_state(
                final,
                target,
                radius=radius,
                target_density=float(config["target_density"]),
                falloff_floor=float(config["target_falloff_floor"]),
            )
    else:
        target_norm, com_loss, balanced_loss, target_mask_loss = _target_losses_for_state(
            final,
            target,
            radius=radius,
            target_density=float(config["target_density"]),
            falloff_floor=float(config["target_falloff_floor"]),
        )
    mass_low, mass_high = _window(config, "mass_ratio")
    active_low, active_high = _window(config, "active_ratio")
    compact_low, compact_high = _window(config, "compactness_ratio")
    mass_fraction_low, mass_fraction_high = _window(config, "mass_fraction")
    active_fraction_low, active_fraction_high = _window(config, "active_fraction")
    absolute_occupancy_loss = fraction_window_loss(terms["mass_fraction"], mass_fraction_low, mass_fraction_high) + fraction_window_loss(
        terms["active_fraction_soft"],
        active_fraction_low,
        active_fraction_high,
    )
    losses = {
        "com": com_loss,
        "balanced_target": balanced_loss,
        "target_mask": target_mask_loss,
        "absolute_occupancy": absolute_occupancy_loss,
        "mass_ratio": ratio_window_loss(terms["mass_ratio"], mass_low, mass_high),
        "active_ratio": ratio_window_loss(terms["active_ratio_soft"], active_low, active_high),
        "compactness_ratio": ratio_window_loss(terms["compactness_ratio"], compact_low, compact_high),
        "anisotropy": F.relu(terms["anisotropy"] - float(config["anisotropy_max"])) ** 2,
        "border": terms["border_mass"] ** 2,
        "visibility": F.relu(float(config["min_active_fraction_soft"]) - terms["active_fraction_soft"]) ** 2
        + F.relu(float(config["min_max_density"]) - terms["max_density"]) ** 2,
    }
    total = sum(torch.as_tensor(weights[key], device=final.device, dtype=final.dtype) * value for key, value in losses.items())
    return total, {**terms, **losses, **stage_terms, "target_distance_norm": target_norm, "loss_total": total}


def rescue_unstable_animal_loss(
    states: list[torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
    objective: dict[str, object] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = _deep_update(RESCUE_UNSTABLE_ANIMAL_OBJECTIVE, objective)
    weights = _weights(config, weights)
    mass_low, mass_high = _window(config, "mass_ratio")
    active_low, active_high = _window(config, "active_ratio")
    mass_fraction_low, mass_fraction_high = _window(config, "mass_fraction")
    active_fraction_low, active_fraction_high = _window(config, "active_fraction")
    compact_low, compact_high = _window(config, "compactness_ratio")
    initial = states[0]
    sampled_states = states[1:] or states
    losses = []
    last_terms: dict[str, torch.Tensor] = {}
    for state in sampled_states:
        terms = _base_terms(state, initial)
        per_state = {
            "mass_ratio": ratio_window_loss(terms["mass_ratio"], mass_low, mass_high),
            "active_ratio": ratio_window_loss(terms["active_ratio_soft"], active_low, active_high),
            "absolute_occupancy": fraction_window_loss(terms["mass_fraction"], mass_fraction_low, mass_fraction_high)
            + fraction_window_loss(terms["active_fraction_soft"], active_fraction_low, active_fraction_high),
            "compactness_ratio": ratio_window_loss(terms["compactness_ratio"], compact_low, compact_high),
            "anisotropy": F.relu(terms["anisotropy"] - float(config["anisotropy_max"])) ** 2,
            "border": terms["border_mass"] ** 2,
            "visibility": F.relu(float(config["min_active_fraction_soft"]) - terms["active_fraction_soft"]) ** 2
            + F.relu(float(config["min_max_density"]) - terms["max_density"]) ** 2,
        }
        losses.append(sum(torch.as_tensor(weights[key], device=state.device, dtype=state.dtype) * value for key, value in per_state.items()))
        last_terms = {**terms, **per_state}
    total = torch.stack(losses).mean()
    return total, {**last_terms, "loss_total": total}


PROFILE_LOSSES: dict[str, Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]] = {
    "move_shape_target": move_shape_target_loss,
    "rescue_unstable_animal": rescue_unstable_animal_loss,
}


def terms_to_json(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in terms.items()}
