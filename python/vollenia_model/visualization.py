from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .kernels import spatial_kernel
from .spec import KernelSpec, ModelSpec


def _cpu_numpy(value: torch.Tensor):
    return value.detach().float().cpu().numpy()


def _radial_profile(kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nz, ny, nx = kernel.shape
    z = torch.arange(nz, device=kernel.device, dtype=kernel.dtype)
    y = torch.arange(ny, device=kernel.device, dtype=kernel.dtype)
    x = torch.arange(nx, device=kernel.device, dtype=kernel.dtype)
    z = torch.minimum(z, torch.as_tensor(nz, device=kernel.device, dtype=kernel.dtype) - z)
    y = torch.minimum(y, torch.as_tensor(ny, device=kernel.device, dtype=kernel.dtype) - y)
    x = torch.minimum(x, torch.as_tensor(nx, device=kernel.device, dtype=kernel.dtype) - x)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    bins = torch.floor(torch.sqrt(xx * xx + yy * yy + zz * zz)).long()
    max_bin = int(bins.max().item())
    sums = torch.zeros(max_bin + 1, device=kernel.device, dtype=kernel.dtype)
    counts = torch.zeros(max_bin + 1, device=kernel.device, dtype=kernel.dtype)
    sums.scatter_add_(0, bins.reshape(-1), kernel.reshape(-1))
    counts.scatter_add_(0, bins.reshape(-1), torch.ones_like(kernel).reshape(-1))
    radius = torch.arange(max_bin + 1, device=kernel.device, dtype=kernel.dtype)
    return radius, sums / torch.clamp_min(counts, 1.0)


def plot_kernel(spec: ModelSpec, kernel: KernelSpec, out_dir: Path, index: int, *, device: str = "cuda") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    spatial = spatial_kernel(spec.dims, kernel, device=device)
    centered = spatial_kernel(spec.dims, kernel, device=device, centered=True)
    z_mid = centered.shape[0] // 2

    slice_path = out_dir / f"kernel_{index:02d}_{kernel.name}_slice.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(_cpu_numpy(centered[z_mid]), cmap="viridis")
    plt.colorbar()
    plt.title(f"{kernel.name} z-mid")
    plt.tight_layout()
    plt.savefig(slice_path, dpi=150)
    plt.close()
    paths.append(slice_path)

    radius, profile = _radial_profile(spatial)
    profile_path = out_dir / f"kernel_{index:02d}_{kernel.name}_radial_profile.png"
    plt.figure(figsize=(5, 3))
    plt.plot(_cpu_numpy(radius), _cpu_numpy(profile))
    plt.title(f"{kernel.name} radial profile")
    plt.xlabel("radius bin")
    plt.ylabel("mean value")
    plt.tight_layout()
    plt.savefig(profile_path, dpi=150)
    plt.close()
    paths.append(profile_path)

    basis_path = out_dir / f"kernel_{index:02d}_{kernel.name}_basis_components.png"
    q = torch.linspace(0.0, 1.5, 256, device=device)
    plt.figure(figsize=(5, 3))
    if kernel.family == "smooth_gaussian_mixture" and kernel.basis:
        envelope = torch.sigmoid(float(kernel.envelope_sharpness) * (1.0 - q))
        total = torch.zeros_like(q)
        for b, basis in enumerate(kernel.basis):
            width = max(float(basis.width), 1.0e-6)
            component = float(basis.amplitude) * torch.exp(-0.5 * ((q - float(basis.center)) / width).square()) * envelope
            total = total + component
            plt.plot(_cpu_numpy(q), _cpu_numpy(component), label=f"basis {b}")
        plt.plot(_cpu_numpy(q), _cpu_numpy(total), label="sum", linewidth=2)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No smooth basis", ha="center", va="center")
    plt.title(f"{kernel.name} basis components")
    plt.xlabel("q = radius / R")
    plt.tight_layout()
    plt.savefig(basis_path, dpi=150)
    plt.close()
    paths.append(basis_path)
    return paths


def plot_model_kernels(spec: ModelSpec, out_dir: str | Path, *, device: str = "cuda") -> list[Path]:
    out_path = Path(out_dir)
    all_paths: list[Path] = []
    for i, kernel in enumerate(spec.kernels):
        all_paths.extend(plot_kernel(spec, kernel, out_path, i, device=device))
    return all_paths
