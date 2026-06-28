from __future__ import annotations

from dataclasses import dataclass

import torch

from .spec import KernelSpec, ModelSpec


@dataclass
class KernelBank:
    spatial: torch.Tensor
    spectrum: torch.Tensor


def _wrapped_distance_grid(
    dims: tuple[int, int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    centered: bool = False,
) -> torch.Tensor:
    nx, ny, nz = dims
    z = torch.arange(nz, device=device, dtype=dtype)
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    if centered:
        z = z - nz // 2
        y = y - ny // 2
        x = x - nx // 2
    else:
        z = torch.minimum(z, torch.as_tensor(nz, device=device, dtype=dtype) - z)
        y = torch.minimum(y, torch.as_tensor(ny, device=device, dtype=dtype) - y)
        x = torch.minimum(x, torch.as_tensor(nx, device=device, dtype=dtype) - x)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.sqrt(xx * xx + yy * yy + zz * zz)


def spatial_kernel(
    dims: tuple[int, int, int],
    kernel: KernelSpec,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    centered: bool = False,
) -> torch.Tensor:
    distance = _wrapped_distance_grid(dims, device=device, dtype=dtype, centered=centered)
    q = distance / max(float(kernel.radius), 1.0e-6)
    if kernel.family == "legacy_shell":
        if kernel.shell_weights:
            weights = torch.tensor(kernel.shell_weights, device=device, dtype=dtype)
            shell_position = q * float(len(kernel.shell_weights))
            shell_index = torch.clamp(torch.floor(shell_position).long(), 0, len(kernel.shell_weights) - 1)
            local = shell_position - torch.floor(shell_position)
            core = (4.0 * local * (1.0 - local)).clamp_min(0.0).pow(4)
            values = torch.where((q >= 0.0) & (q < 1.0), weights[shell_index] * core, torch.zeros_like(q))
        else:
            core = (4.0 * q * (1.0 - q)).clamp_min(0.0).pow(4)
            values = torch.where((q >= 0.0) & (q < 1.0), core, torch.zeros_like(q))
    elif kernel.family == "smooth_gaussian_mixture":
        envelope = torch.sigmoid(float(kernel.envelope_sharpness) * (1.0 - q))
        raw = torch.zeros_like(q)
        basis = kernel.basis or []
        for entry in basis:
            width = max(float(entry.width), 1.0e-6)
            diff = (q - float(entry.center)) / width
            raw = raw + float(entry.amplitude) * torch.exp(-0.5 * diff * diff)
        values = envelope * raw
    else:
        raise ValueError(f"Unsupported kernel family: {kernel.family}")
    total = values.sum()
    if torch.abs(total).item() <= 1.0e-12:
        raise ValueError(f"Kernel sum is zero and cannot be normalized: {kernel.name}")
    return values / total


def build_kernel_bank(
    spec: ModelSpec,
    dims: tuple[int, int, int] | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> KernelBank:
    dims = dims or spec.dims
    spatial = torch.stack(
        [spatial_kernel(dims, kernel, device=device, dtype=dtype) for kernel in spec.kernels],
        dim=0,
    )
    spectrum = torch.fft.rfftn(spatial, dim=(-3, -2, -1))
    return KernelBank(spatial=spatial, spectrum=spectrum)
