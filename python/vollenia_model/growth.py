from __future__ import annotations

import torch

from .spec import GrowthSpec


def growth(u: torch.Tensor, spec: GrowthSpec) -> torch.Tensor:
    sigma = max(float(spec.sigma), 1.0e-6)
    diff = u - float(spec.mu)
    if spec.family == "gaussian":
        return 2.0 * torch.exp(-0.5 * (diff * diff) / (sigma * sigma)) - 1.0
    if spec.family == "polynomial_lenia3d":
        value = torch.clamp(1.0 - (diff * diff) / (9.0 * sigma * sigma), min=0.0)
        return 2.0 * value.square().square() - 1.0
    raise ValueError(f"Unsupported growth family: {spec.family}")
