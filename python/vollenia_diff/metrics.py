from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


def _cache_key(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> tuple[object, ...]:
    return tuple(shape), str(device), str(dtype)


@lru_cache(maxsize=64)
def _coords_cached(shape: tuple[int, ...], device_text: str, dtype_text: str) -> torch.Tensor:
    device = torch.device(device_text)
    dtype = getattr(torch, dtype_text.split(".")[-1])
    axes = [torch.arange(size, device=device, dtype=dtype) for size in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grids, dim=0)


@lru_cache(maxsize=64)
def _border_mask_cached(shape: tuple[int, ...], width: int, device_text: str) -> torch.Tensor:
    device = torch.device(device_text)
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    for axis, size in enumerate(shape):
        index = torch.arange(size, device=device)
        axis_mask = (index < width) | (index >= size - width)
        view_shape = [1 for _ in shape]
        view_shape[axis] = size
        mask = mask | axis_mask.reshape(view_shape)
    return mask


def coordinate_grid(state: torch.Tensor) -> torch.Tensor:
    key = _cache_key(tuple(int(v) for v in state.shape), state.device, state.dtype)
    return _coords_cached(key[0], key[1], key[2])


def mass(state: torch.Tensor) -> torch.Tensor:
    return state.sum()


def mean_density(state: torch.Tensor) -> torch.Tensor:
    return state.mean()


def max_density(state: torch.Tensor) -> torch.Tensor:
    return state.max()


def center_of_mass(state: torch.Tensor, coords: torch.Tensor | None = None, eps: float = 1.0e-8) -> torch.Tensor:
    coords = coordinate_grid(state) if coords is None else coords
    m = mass(state)
    flat_state = state.reshape(-1)
    flat_coords = coords.reshape(coords.shape[0], -1)
    return (flat_coords * flat_state.unsqueeze(0)).sum(dim=1) / (m + eps)


def second_moment(
    state: torch.Tensor,
    coords: torch.Tensor | None = None,
    com: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    coords = coordinate_grid(state) if coords is None else coords
    com = center_of_mass(state, coords, eps=eps) if com is None else com
    centered = coords - com.reshape((-1,) + (1,) * state.ndim)
    radius2 = (centered * centered).sum(dim=0)
    return (state * radius2).sum() / (mass(state) + eps)


def covariance(
    state: torch.Tensor,
    coords: torch.Tensor | None = None,
    com: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    coords = coordinate_grid(state) if coords is None else coords
    com = center_of_mass(state, coords, eps=eps) if com is None else com
    flat_state = state.reshape(-1)
    flat_coords = coords.reshape(coords.shape[0], -1)
    centered = flat_coords - com.reshape(-1, 1)
    return (centered * flat_state.reshape(1, -1)) @ centered.transpose(0, 1) / (mass(state) + eps)


def covariance_eigvals(state: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    cov = covariance(state, eps=eps)
    return torch.linalg.eigvalsh(cov)


def anisotropy(state: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    eigvals = covariance_eigvals(state, eps=eps)
    return eigvals[-1] / (eigvals[0] + eps)


def target_distance(state: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    target = target.to(device=state.device, dtype=state.dtype)
    return torch.linalg.vector_norm(center_of_mass(state, eps=eps) - target)


def obstacle_overlap(state: torch.Tensor, obstacle: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    obstacle = obstacle.to(device=state.device, dtype=state.dtype)
    return (state * obstacle).sum() / (mass(state) + eps)


def border_mass(state: torch.Tensor, width: int = 2, eps: float = 1.0e-8) -> torch.Tensor:
    shape = tuple(int(v) for v in state.shape)
    mask = _border_mask_cached(shape, int(width), str(state.device)).to(dtype=state.dtype)
    return (state * mask).sum() / (mass(state) + eps)


def axis_border_mass(state: torch.Tensor, width: int = 2, eps: float = 1.0e-8) -> torch.Tensor:
    values: list[torch.Tensor] = []
    total = mass(state) + eps
    for axis, size in enumerate(state.shape):
        index = torch.arange(int(size), device=state.device)
        axis_mask = (index < int(width)) | (index >= int(size) - int(width))
        view_shape = [1 for _ in state.shape]
        view_shape[axis] = int(size)
        mask = axis_mask.reshape(view_shape).to(dtype=state.dtype)
        values.append((state * mask).sum() / total)
    return torch.stack(values)


def active_voxels(state: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    return (state > threshold).sum()


def density_view(state: torch.Tensor, mode: str = "raw") -> torch.Tensor:
    if mode == "raw":
        return state
    if mode == "clamp":
        return torch.clamp(state, 0.0, 1.0)
    if mode == "sigmoid":
        return torch.sigmoid(state)
    if mode == "softplus":
        return torch.nn.functional.softplus(state)
    raise ValueError(f"Unknown density view mode: {mode}")


def _axis_projection(mask: torch.Tensor, axis: int) -> torch.Tensor:
    dims = tuple(index for index in range(mask.ndim) if index != axis)
    return mask.to(dtype=torch.bool).any(dim=dims)


def _circular_span_norm(projection: torch.Tensor) -> torch.Tensor:
    projection = projection.to(dtype=torch.bool)
    size = int(projection.numel())
    dtype = torch.float32
    if size == 0:
        return torch.zeros((), device=projection.device, dtype=dtype)
    active_index = torch.nonzero(projection, as_tuple=False).flatten()
    active_count = int(active_index.numel())
    if active_count == 0:
        return torch.zeros((), device=projection.device, dtype=dtype)
    if active_count == size:
        return torch.ones((), device=projection.device, dtype=dtype)
    gaps = active_index[1:] - active_index[:-1] - 1
    wrap_gap = active_index[0] + size - active_index[-1] - 1
    max_gap = torch.cat([gaps, wrap_gap.reshape(1)]).max()
    span = torch.as_tensor(size, device=projection.device, dtype=dtype) - max_gap.to(dtype=dtype)
    return span / float(size)


def active_axis_descriptors(state: torch.Tensor, active_threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    active_mask = state > float(active_threshold)
    coverage: list[torch.Tensor] = []
    circular_span: list[torch.Tensor] = []
    for axis, size in enumerate(state.shape):
        projection = _axis_projection(active_mask, axis)
        coverage.append(projection.to(dtype=state.dtype).mean())
        circular_span.append(_circular_span_norm(projection).to(device=state.device, dtype=state.dtype))
    return torch.stack(coverage), torch.stack(circular_span)


def normalized_descriptors(
    state: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    active_threshold: float = 0.05,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    shape = tuple(int(v) for v in state.shape)
    volume_voxels = float(state.numel())
    min_dim = float(min(shape))
    coords = coordinate_grid(state)
    state_mass = mass(state)
    com = center_of_mass(state, coords, eps=eps)
    moment = second_moment(state, coords, com, eps=eps)
    eigvals = torch.linalg.eigvalsh(covariance(state, coords, com, eps=eps))
    active_mask = (state > active_threshold).to(dtype=state.dtype)
    active = active_mask.sum()
    active_moment = second_moment(active_mask, coords, eps=eps)
    active_axis_coverage, active_axis_circular_span = active_axis_descriptors(state, active_threshold=active_threshold)
    full_axis_count = (active_axis_circular_span >= 0.85).to(dtype=state.dtype).sum()
    descriptor: dict[str, torch.Tensor] = {
        "mass_fraction": state_mass / volume_voxels,
        "active_fraction": active / volume_voxels,
        "second_moment_norm": moment / (min_dim * min_dim),
        "body_radius": torch.sqrt(torch.clamp(moment, min=0.0) + eps),
        "active_body_radius": torch.sqrt(torch.clamp(active_moment, min=0.0) + eps),
        "anisotropy": eigvals[-1] / (eigvals[0] + eps),
        "border_mass": border_mass(state, eps=eps),
        "axis_border_mass": axis_border_mass(state, eps=eps),
        "active_axis_coverage_norm": active_axis_coverage,
        "active_axis_circular_span_norm": active_axis_circular_span,
        "full_axis_count": full_axis_count,
    }
    descriptor["body_radius_norm"] = descriptor["body_radius"] / min_dim
    descriptor["active_body_radius_norm"] = descriptor["active_body_radius"] / min_dim
    shape_tensor = torch.tensor(
        [max(size - 1, 1) for size in shape],
        device=state.device,
        dtype=state.dtype,
    )
    descriptor["com_norm"] = com / shape_tensor
    if target is not None:
        distance = target_distance(state, target, eps=eps)
        descriptor["target_distance_body"] = distance / (descriptor["body_radius"] + eps)
        descriptor["target_distance_norm"] = distance / min_dim
    if initial is not None:
        initial_mass = mass(initial)
        initial_active = active_voxels(initial, threshold=active_threshold).to(dtype=state.dtype)
        initial_moment = second_moment(initial, eps=eps)
        descriptor["mass_ratio"] = state_mass / (initial_mass + eps)
        descriptor["active_ratio"] = active / (initial_active + eps)
        descriptor["compactness_ratio"] = moment / (initial_moment + eps)
    return descriptor


def descriptors_to_json(descriptors: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in descriptors.items():
        detached = value.detach().cpu()
        if detached.ndim == 0:
            result[key] = float(detached)
        else:
            result[key] = [float(v) for v in detached.reshape(-1).tolist()]
    return result


def state_summary(state: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, Any]:
    with torch.no_grad():
        coords = coordinate_grid(state)
        com = center_of_mass(state, coords)
        cov = covariance(state, coords, com)
        eigvals = torch.linalg.eigvalsh(cov)
        summary: dict[str, Any] = {
            "mass": float(mass(state).detach().cpu()),
            "mean_density": float(mean_density(state).detach().cpu()),
            "max_density": float(max_density(state).detach().cpu()),
            "active_voxels": int(active_voxels(state).detach().cpu()),
            "center_of_mass": [float(v) for v in com.detach().cpu().tolist()],
            "second_moment": float(second_moment(state, coords, com).detach().cpu()),
            "covariance_eigvals": [float(v) for v in eigvals.detach().cpu().tolist()],
            "anisotropy": float((eigvals[-1] / (eigvals[0] + 1.0e-8)).detach().cpu()),
            "border_mass": float(border_mass(state).detach().cpu()),
        }
        descriptors = normalized_descriptors(state, target=target)
        summary["mass_fraction"] = float(descriptors["mass_fraction"].detach().cpu())
        summary["active_fraction"] = float(descriptors["active_fraction"].detach().cpu())
        summary["body_radius_norm"] = float(descriptors["body_radius_norm"].detach().cpu())
        summary["active_body_radius_norm"] = float(descriptors["active_body_radius_norm"].detach().cpu())
        summary["axis_border_mass"] = [float(v) for v in descriptors["axis_border_mass"].detach().cpu().reshape(-1).tolist()]
        summary["active_axis_coverage_norm"] = [float(v) for v in descriptors["active_axis_coverage_norm"].detach().cpu().reshape(-1).tolist()]
        summary["active_axis_circular_span_norm"] = [
            float(v) for v in descriptors["active_axis_circular_span_norm"].detach().cpu().reshape(-1).tolist()
        ]
        summary["full_axis_count"] = int(descriptors["full_axis_count"].detach().cpu())
        if target is not None:
            summary["target_distance"] = float(target_distance(state, target).detach().cpu())
        return summary


def rollout_summary(states: list[torch.Tensor], target: torch.Tensor | None = None) -> dict[str, Any]:
    if not states:
        return {"states": []}
    summaries = [state_summary(state, target) for state in states]
    masses = torch.tensor([item["mass"] for item in summaries], dtype=torch.float64)
    return {
        "states": summaries,
        "mass_mean": float(masses.mean()),
        "mass_std": float(masses.std(unbiased=False)),
        "mass_cv": float(masses.std(unbiased=False) / (masses.mean() + 1.0e-8)),
    }
