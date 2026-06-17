from __future__ import annotations

from functools import lru_cache

import torch

from .params import LeniaParams


def kernel_core(r: torch.Tensor, kn: int) -> torch.Tensor:
    """Evaluate the Lenia3D/C++ kernel core functions on local shell radius r."""

    r = r.to(dtype=torch.float32) if not r.is_floating_point() else r
    out = torch.zeros_like(r)
    inside = (r >= 0.0) & (r <= 1.0)
    if kn == 2:
        valid = inside & (r > 0.0) & (r < 1.0)
        rv = r[valid]
        out[valid] = torch.exp(4.0 - 1.0 / (rv * (1.0 - rv)))
    elif kn == 3:
        out[(r >= 0.25) & (r <= 0.75)] = 1.0
    elif kn == 4:
        out[(r >= 0.25) & (r <= 0.75)] = 1.0
        out[(r >= 0.0) & (r < 0.25)] = 0.5
    else:
        value = 4.0 * r[inside] * (1.0 - r[inside])
        out[inside] = value**4
    return out


def growth_function(u: torch.Tensor, params: LeniaParams) -> torch.Tensor:
    params = params.sanitized()
    diff = u - params.m
    sigma = max(params.s, 1.0e-5)
    if params.gn == 2:
        return 2.0 * torch.exp(-(diff * diff) / (2.0 * sigma * sigma)) - 1.0
    if params.gn == 3:
        return torch.where(torch.abs(diff) <= sigma, torch.ones_like(u), -torch.ones_like(u))
    value = torch.clamp(1.0 - (diff * diff) / (9.0 * sigma * sigma), min=0.0)
    return 2.0 * value**4 - 1.0


@lru_cache(maxsize=64)
def _wrapped_axes(shape: tuple[int, ...], device_text: str, dtype_text: str) -> tuple[torch.Tensor, ...]:
    device = torch.device(device_text)
    dtype = getattr(torch, dtype_text.split(".")[-1])
    axes: list[torch.Tensor] = []
    for size in shape:
        axis = torch.arange(size, device=device, dtype=dtype)
        axes.append(torch.minimum(axis, torch.tensor(float(size), device=device, dtype=dtype) - axis))
    return tuple(axes)


def wrapped_distance(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(shape) not in (2, 3):
        raise ValueError(f"Expected 2D or 3D shape, got {shape}")
    axes = _wrapped_axes(tuple(int(v) for v in shape), str(device), str(dtype))
    grids = torch.meshgrid(*axes, indexing="ij")
    squared = torch.zeros(shape, device=device, dtype=dtype)
    for grid in grids:
        squared = squared + grid * grid
    return torch.sqrt(squared)


def build_radial_kernel(
    shape: tuple[int, ...],
    params: LeniaParams,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized wrapped radial shell kernel for [Y, X] or [Z, Y, X]."""

    params = params.sanitized()
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    shape = tuple(int(v) for v in shape)
    distance = wrapped_distance(shape, device, dtype)
    q = distance / params.R
    kernel = torch.zeros(shape, device=device, dtype=dtype)
    mask = q < 1.0
    if mask.any():
        shell_weights = torch.tensor(params.b, device=device, dtype=dtype)
        shell_count = int(shell_weights.numel())
        shell_position = q[mask] * float(shell_count)
        shell_floor = torch.floor(shell_position)
        shell_index = torch.clamp(shell_floor.to(torch.long), 0, shell_count - 1)
        local_r = shell_position - shell_floor
        kernel[mask] = shell_weights[shell_index] * kernel_core(local_r, params.kn).to(dtype=dtype)
    total = kernel.sum()
    if torch.any(total <= 0):
        raise ValueError("Lenia kernel sum is zero; cannot normalize")
    return kernel / total


def kernel_spectrum(kernel: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(kernel.ndim))
    return torch.fft.rfftn(kernel, dim=dims)
