from __future__ import annotations

import math

import torch


def transport_dd(dt: float, flow_max: float, transport_sigma: float = 0.5, reintegration_dd: int = 0) -> int:
    if reintegration_dd > 0:
        return int(reintegration_dd)
    return max(1, int(math.ceil(max(0.0, dt * flow_max) + transport_sigma - 1.0e-6)))


def sobel3d(field: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)
    gz = torch.zeros_like(field)
    for oz in (-1, 0, 1):
        wz = 2.0 if oz == 0 else 1.0
        for oy in (-1, 0, 1):
            wy = 2.0 if oy == 0 else 1.0
            for ox in (-1, 0, 1):
                wx = 2.0 if ox == 0 else 1.0
                rolled = torch.roll(field, shifts=(oz, oy, ox), dims=(-3, -2, -1))
                gx = gx + float(ox) * wy * wz * rolled
                gy = gy + float(oy) * wx * wz * rolled
                gz = gz + float(oz) * wx * wy * rolled
    return torch.stack((gx / 32.0, gy / 32.0, gz / 32.0), dim=0)


def reintegrate_sigma_half(
    state: torch.Tensor,
    flow: torch.Tensor,
    *,
    dt: float,
    flow_max: float,
    border: str = "torus",
    reintegration_dd: int = 0,
) -> torch.Tensor:
    if border not in {"torus", "wall"}:
        raise ValueError(f"Unsupported border: {border}")
    sigma = 0.5
    dd = transport_dd(dt, flow_max, sigma, reintegration_dd)
    ma = max(float(dd) - sigma, 0.0)
    nz, ny, nx = state.shape
    device = state.device
    dtype = state.dtype
    z = torch.arange(nz, device=device, dtype=dtype).view(nz, 1, 1) + 0.5
    y = torch.arange(ny, device=device, dtype=dtype).view(1, ny, 1) + 0.5
    x = torch.arange(nx, device=device, dtype=dtype).view(1, 1, nx) + 0.5
    total = torch.zeros_like(state)
    for oz in range(-dd, dd + 1):
        for oy in range(-dd, dd + 1):
            for ox in range(-dd, dd + 1):
                source = torch.roll(state, shifts=(oz, oy, ox), dims=(-3, -2, -1))
                source_flow = torch.roll(flow, shifts=(oz, oy, ox), dims=(-3, -2, -1))
                sz = torch.remainder(z - float(oz) - 0.5, float(nz)) + 0.5
                sy = torch.remainder(y - float(oy) - 0.5, float(ny)) + 0.5
                sx = torch.remainder(x - float(ox) - 0.5, float(nx)) + 0.5
                mu_z = sz + torch.clamp(dt * source_flow[2], -ma, ma)
                mu_y = sy + torch.clamp(dt * source_flow[1], -ma, ma)
                mu_x = sx + torch.clamp(dt * source_flow[0], -ma, ma)
                if border == "wall":
                    mu_z = torch.clamp(mu_z, 0.5, float(nz) - 0.5)
                    mu_y = torch.clamp(mu_y, 0.5, float(ny) - 0.5)
                    mu_x = torch.clamp(mu_x, 0.5, float(nx) - 0.5)
                    dz = torch.abs(z - mu_z)
                    dy = torch.abs(y - mu_y)
                    dx = torch.abs(x - mu_x)
                    in_bounds = ((z - float(oz)) >= 0.5) & ((z - float(oz)) <= float(nz) - 0.5)
                    in_bounds = in_bounds & ((y - float(oy)) >= 0.5) & ((y - float(oy)) <= float(ny) - 0.5)
                    in_bounds = in_bounds & ((x - float(ox)) >= 0.5) & ((x - float(ox)) <= float(nx) - 0.5)
                else:
                    dz = torch.minimum(torch.abs(z - mu_z), torch.as_tensor(float(nz), device=device, dtype=dtype) - torch.abs(z - mu_z))
                    dy = torch.minimum(torch.abs(y - mu_y), torch.as_tensor(float(ny), device=device, dtype=dtype) - torch.abs(y - mu_y))
                    dx = torch.minimum(torch.abs(x - mu_x), torch.as_tensor(float(nx), device=device, dtype=dtype) - torch.abs(x - mu_x))
                    in_bounds = True
                weight = torch.relu(1.0 - dx) * torch.relu(1.0 - dy) * torch.relu(1.0 - dz)
                if border == "wall":
                    weight = torch.where(in_bounds, weight, torch.zeros_like(weight))
                total = total + source * weight
    return torch.clamp_min(total, 0.0)
