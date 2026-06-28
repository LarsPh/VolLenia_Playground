from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from vollenia_diff import metrics
from vollenia_diff.rollout_losses import target_from_initial_offset


@dataclass(slots=True)
class TargetStage:
    at_step: int | None = None
    at_step_fraction: float | None = None
    offset_norm_zyx: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    weight_scale: dict[str, float] = field(default_factory=dict)
    target_zyx: list[float] = field(default_factory=list)
    target_norm_zyx: list[float] = field(default_factory=list)


@dataclass(slots=True)
class TargetSpec:
    mode: Literal["initial_offset", "absolute_norm", "absolute_voxel", "none"] = "initial_offset"
    offset_norm_zyx: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.12])
    stages: list[TargetStage] = field(default_factory=list)


@dataclass(slots=True)
class TargetContext:
    spec: TargetSpec
    target: torch.Tensor | None
    target_norm_zyx: list[float] | None
    stages: list[TargetStage] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.spec.mode,
            "offset_norm_zyx": self.spec.offset_norm_zyx,
            "target_zyx": None if self.target is None else [float(v) for v in self.target.detach().cpu().reshape(-1).tolist()],
            "target_norm_zyx": self.target_norm_zyx,
            "stages": [
                {
                    "at_step": stage.at_step,
                    "at_step_fraction": stage.at_step_fraction,
                    "offset_norm_zyx": stage.offset_norm_zyx,
                    "weight_scale": stage.weight_scale,
                    "target_zyx": stage.target_zyx,
                    "target_norm_zyx": stage.target_norm_zyx,
                }
                for stage in self.stages
            ],
        }


def _target_norm(state: torch.Tensor, target: torch.Tensor | None) -> list[float] | None:
    if target is None:
        return None
    denom = torch.tensor([max(size - 1, 1) for size in state.shape], device=target.device, dtype=target.dtype)
    return [float(v) for v in (target / denom).detach().cpu().reshape(-1).tolist()]


def _target_from_spec(state: torch.Tensor, spec: TargetSpec, offset: list[float] | None = None) -> torch.Tensor | None:
    if spec.mode == "none":
        return None
    if spec.mode == "initial_offset":
        return target_from_initial_offset(state, offset or spec.offset_norm_zyx)
    if spec.mode == "absolute_norm":
        values = offset or spec.offset_norm_zyx
        shape = torch.tensor([max(size - 1, 1) for size in state.shape], device=state.device, dtype=state.dtype)
        return torch.tensor([float(value) for value in values], device=state.device, dtype=state.dtype) * shape
    if spec.mode == "absolute_voxel":
        values = offset or spec.offset_norm_zyx
        return torch.tensor([float(value) for value in values], device=state.device, dtype=state.dtype)
    raise ValueError(f"Unsupported target mode: {spec.mode}")


def target_spec_from_objective(profile: str, objective: dict[str, Any] | None) -> TargetSpec:
    if profile != "move_shape_target":
        return TargetSpec(mode="none", offset_norm_zyx=[0.0, 0.0, 0.0])
    objective = objective or {}
    target_profile = objective.get("target_profile")
    stages: list[TargetStage] = []
    if isinstance(target_profile, dict) and target_profile.get("mode") == "staged_offsets":
        for raw_stage in list(target_profile.get("stages", [])):
            if not isinstance(raw_stage, dict):
                continue
            stages.append(
                TargetStage(
                    at_step=int(raw_stage["at_step"]) if raw_stage.get("at_step") is not None else None,
                    at_step_fraction=float(raw_stage["at_step_fraction"]) if raw_stage.get("at_step_fraction") is not None else None,
                    offset_norm_zyx=[float(value) for value in raw_stage.get("offset_norm_zyx", objective.get("target_offset_norm_zyx", [0.0, 0.0, 0.12]))],
                    weight_scale={key: float(value) for key, value in dict(raw_stage.get("weight_scale", {})).items()},
                )
            )
    return TargetSpec(
        mode=str(objective.get("target_mode", "initial_offset")),  # type: ignore[arg-type]
        offset_norm_zyx=[float(value) for value in objective.get("target_offset_norm_zyx", [0.0, 0.0, 0.12])],
        stages=stages,
    )


def make_target_context(profile: str, state0: torch.Tensor, objective: dict[str, Any] | None) -> TargetContext:
    spec = target_spec_from_objective(profile, objective)
    target = _target_from_spec(state0, spec)
    stages: list[TargetStage] = []
    for stage in spec.stages:
        stage_target = _target_from_spec(state0, spec, stage.offset_norm_zyx)
        stages.append(
            TargetStage(
                at_step=stage.at_step,
                at_step_fraction=stage.at_step_fraction,
                offset_norm_zyx=list(stage.offset_norm_zyx),
                weight_scale=dict(stage.weight_scale),
                target_zyx=[] if stage_target is None else [float(v) for v in stage_target.detach().cpu().reshape(-1).tolist()],
                target_norm_zyx=[] if stage_target is None else (_target_norm(state0, stage_target) or []),
            )
        )
    return TargetContext(spec=spec, target=target, target_norm_zyx=_target_norm(state0, target), stages=stages)


def target_from_context(context: TargetContext | None) -> torch.Tensor | None:
    return None if context is None else context.target
