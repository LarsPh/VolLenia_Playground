from __future__ import annotations

import torch
import torch.nn.functional as F

from . import metrics


def target_loss(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return metrics.target_distance(state, target) ** 2


def mass_range_loss(state: torch.Tensor, mass_min: float, mass_max: float) -> torch.Tensor:
    m = metrics.mass(state)
    return F.relu(torch.as_tensor(mass_min, device=state.device, dtype=state.dtype) - m) ** 2 + F.relu(
        m - torch.as_tensor(mass_max, device=state.device, dtype=state.dtype)
    ) ** 2


def compactness_loss(state: torch.Tensor, max_second_moment: float) -> torch.Tensor:
    moment = metrics.second_moment(state)
    limit = torch.as_tensor(max_second_moment, device=state.device, dtype=state.dtype)
    return F.relu(moment - limit) ** 2


def obstacle_loss(state: torch.Tensor, obstacle: torch.Tensor) -> torch.Tensor:
    return metrics.obstacle_overlap(state, obstacle)


def border_loss(state: torch.Tensor, width: int = 2) -> torch.Tensor:
    return metrics.border_mass(state, width=width)


def toy_loss(
    state: torch.Tensor,
    target: torch.Tensor,
    *,
    mass_min: float,
    mass_max: float,
    max_second_moment: float,
    lambda_mass: float = 1.0e-4,
    lambda_compact: float = 1.0e-4,
    lambda_border: float = 1.0,
) -> torch.Tensor:
    return (
        target_loss(state, target)
        + lambda_mass * mass_range_loss(state, mass_min, mass_max)
        + lambda_compact * compactness_loss(state, max_second_moment)
        + lambda_border * border_loss(state)
    )
