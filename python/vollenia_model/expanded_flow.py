from __future__ import annotations

from dataclasses import dataclass

import torch

from .flow_transport import reintegrate_sigma_half, sobel3d
from .growth import growth
from .kernels import KernelBank, build_kernel_bank
from .spec import ModelSpec


@dataclass
class ModelSpecSimulator:
    spec: ModelSpec
    dims: tuple[int, int, int] | None = None
    device: torch.device | str = "cuda"
    dtype: torch.dtype = torch.float32
    kernel_batch_size: int = 16
    kernel_bank: KernelBank | None = None

    def __post_init__(self) -> None:
        self.dims = self.dims or self.spec.dims
        self.kernel_bank = build_kernel_bank(self.spec, self.dims, device=self.device, dtype=self.dtype)

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.dims or self.spec.dims
        return nz, ny, nx

    def affinity(self, state: torch.Tensor) -> torch.Tensor:
        if self.kernel_bank is None:
            raise RuntimeError("Kernel bank was not initialized")
        u = torch.zeros_like(state)
        state_hat = torch.fft.rfftn(state, dim=(-3, -2, -1))
        kernels = self.spec.kernels
        for start in range(0, len(kernels), self.kernel_batch_size):
            end = min(start + self.kernel_batch_size, len(kernels))
            chunk = kernels[start:end]
            src = torch.tensor([kernel.src for kernel in chunk], device=state.device, dtype=torch.long)
            p_hat = state_hat.index_select(0, src) * self.kernel_bank.spectrum[start:end]
            p = torch.fft.irfftn(p_hat, s=self.spatial_shape, dim=(-3, -2, -1))
            for local_i, kernel in enumerate(chunk):
                if self.spec.channels[kernel.dst].role != "matter":
                    continue
                u[kernel.dst] = u[kernel.dst] + float(kernel.weight) * growth(p[local_i], kernel.growth)
        return u

    def step(self, state: torch.Tensor) -> torch.Tensor:
        u = self.affinity(state)
        if self.spec.update_mode == "flow":
            return self.flow_step(state, u)
        nxt = state + float(self.spec.dt) * u
        return torch.clamp(nxt, 0.0, 1.0) if self.spec.hard_clip else nxt

    def flow_step(self, state: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        matter = self.spec.matter_indices
        a_sum = state[matter].sum(dim=0) if matter else torch.zeros(self.spatial_shape, device=state.device, dtype=state.dtype)
        grad_a = sobel3d(a_sum)
        alpha = torch.clamp((a_sum / max(float(self.spec.flow.theta_A), 1.0e-6)).pow(float(self.spec.flow.alpha_power)), 0.0, 1.0)
        nxt = state.clone()
        for c in range(self.spec.channel_count):
            if c not in matter:
                continue
            grad_u = sobel3d(affinity[c])
            flow = (1.0 - alpha).unsqueeze(0) * grad_u - alpha.unsqueeze(0) * grad_a
            flow = torch.clamp(flow, -float(self.spec.flow.flow_max), float(self.spec.flow.flow_max))
            nxt[c] = reintegrate_sigma_half(
                state[c],
                flow,
                dt=float(self.spec.dt),
                flow_max=float(self.spec.flow.flow_max),
                border=self.spec.flow.border,
                reintegration_dd=int(self.spec.flow.reintegration_dd),
            )
        return torch.clamp(nxt, 0.0, 1.0) if self.spec.hard_clip else nxt


def seed_state(
    spec: ModelSpec,
    dims: tuple[int, int, int] | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 1,
) -> torch.Tensor:
    nx, ny, nz = dims or spec.dims
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    z = torch.linspace(-1.0, 1.0, nz, device=device, dtype=dtype).view(nz, 1, 1)
    y = torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype).view(1, ny, 1)
    x = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype).view(1, 1, nx)
    channels = []
    for c in range(spec.channel_count):
        cx = -0.18 + 0.12 * c
        cy = 0.10 * torch.sin(torch.tensor(float(seed % 17 + c), device=device, dtype=dtype))
        cz = 0.10 * torch.cos(torch.tensor(float(seed % 11 + c), device=device, dtype=dtype))
        d2 = (x - cx).square() + (y - cy).square() + (z - cz).square()
        shell_r = torch.sqrt(x.square() + y.square() + z.square())
        value = 0.85 * torch.exp(-d2 / (2.0 * 0.18 * 0.18))
        value = value + 0.18 * torch.exp(-((shell_r - 0.42) / 0.08).square())
        if spec.channels[c].role != "matter":
            value = torch.zeros_like(value)
        channels.append(torch.clamp(value, 0.0, 1.0))
    return torch.stack(channels, dim=0)


def rollout(simulator: ModelSpecSimulator, state: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(max(0, int(steps))):
        state = simulator.step(state)
    return state
