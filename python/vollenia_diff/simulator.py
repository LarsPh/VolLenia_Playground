from __future__ import annotations

import torch

from .kernels import build_radial_kernel, kernel_spectrum
from .params import LeniaParams


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VolLenia PyTorch backend is GPU-only. Install a CUDA-enabled PyTorch wheel "
            "with `uv sync --refresh` and verify torch.cuda.is_available()."
        )


def _clip_mode_id(clip_mode: str) -> int:
    if clip_mode == "hard":
        return 1
    if clip_mode == "none":
        return 0
    raise ValueError(f"Unknown clip_mode: {clip_mode}")


def _growth_from_id(u: torch.Tensor, m: float, s: float, gn: int) -> torch.Tensor:
    diff = u - m
    sigma = max(float(s), 1.0e-5)
    if gn == 2:
        return 2.0 * torch.exp(-(diff * diff) / (2.0 * sigma * sigma)) - 1.0
    if gn == 3:
        return torch.where(torch.abs(diff) <= sigma, torch.ones_like(u), -torch.ones_like(u))
    value = torch.clamp(1.0 - (diff * diff) / (9.0 * sigma * sigma), min=0.0)
    return 2.0 * value**4 - 1.0


def lenia_step_tensor(
    state: torch.Tensor,
    kernel_hat: torch.Tensor,
    shape: tuple[int, ...],
    m: float,
    s: float,
    T: float,
    gn: int,
    clip_mode_id: int,
) -> torch.Tensor:
    """Compile-friendly single Lenia step. Keep logging/export outside this function."""

    dims = tuple(range(state.ndim))
    state_hat = torch.fft.rfftn(state, dim=dims)
    u = torch.fft.irfftn(state_hat * kernel_hat, s=shape, dim=dims)
    next_state = state + _growth_from_id(u, m, s, gn) / max(float(T), 1.0)
    if clip_mode_id == 1:
        return torch.clamp(next_state, 0.0, 1.0)
    return next_state


class LeniaSimulator:
    """Cached PyTorch FFT simulator for single-channel 2D/3D Lenia."""

    def __init__(
        self,
        shape: tuple[int, ...],
        params: LeniaParams | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        clip_mode: str = "hard",
        compile_step: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
    ) -> None:
        if len(shape) not in (2, 3):
            raise ValueError(f"Expected 2D or 3D shape, got {shape}")
        self.shape = tuple(int(v) for v in shape)
        self.params = (params or LeniaParams()).sanitized()
        self.device = torch.device(device or "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("LeniaSimulator is GPU-only; pass device='cuda'.")
        require_cuda()
        self.dtype = dtype
        self.clip_mode = clip_mode
        self.compile_step = compile_step
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self._kernel: torch.Tensor | None = None
        self._kernel_hat: torch.Tensor | None = None
        self._kernel_key: tuple[object, ...] | None = None
        self._step_callable = None
        self._step_key: tuple[object, ...] | None = None

    def _current_kernel_key(self) -> tuple[object, ...]:
        params = self.params.sanitized()
        return (
            self.shape,
            str(self.device),
            str(self.dtype),
            round(params.R, 8),
            tuple(round(float(v), 8) for v in params.b),
            params.kn,
        )

    @property
    def kernel(self) -> torch.Tensor:
        self._ensure_kernel()
        assert self._kernel is not None
        return self._kernel

    @property
    def kernel_hat(self) -> torch.Tensor:
        self._ensure_kernel()
        assert self._kernel_hat is not None
        return self._kernel_hat

    def _ensure_kernel(self) -> None:
        key = self._current_kernel_key()
        if key == self._kernel_key and self._kernel_hat is not None:
            return
        kernel = build_radial_kernel(self.shape, self.params, device=self.device, dtype=self.dtype)
        self._kernel = kernel
        self._kernel_hat = kernel_spectrum(kernel)
        self._kernel_key = key
        self._step_callable = None
        self._step_key = None

    def _current_step_key(self) -> tuple[object, ...]:
        params = self.params.sanitized()
        return (
            self._current_kernel_key(),
            round(params.m, 8),
            round(params.s, 8),
            round(params.T, 8),
            params.gn,
            self.clip_mode,
            self.compile_step,
            self.compile_backend,
            self.compile_mode,
        )

    def _ensure_step_callable(self) -> None:
        self._ensure_kernel()
        key = self._current_step_key()
        if key == self._step_key and self._step_callable is not None:
            return
        params = self.params.sanitized()
        clip_id = _clip_mode_id(self.clip_mode)

        def step_fn(state: torch.Tensor, kernel_hat: torch.Tensor) -> torch.Tensor:
            return lenia_step_tensor(state, kernel_hat, self.shape, params.m, params.s, params.T, params.gn, clip_id)

        if self.compile_step:
            self._step_callable = torch.compile(step_fn, backend=self.compile_backend, mode=self.compile_mode)
        else:
            self._step_callable = step_fn
        self._step_key = key

    def potential(self, state: torch.Tensor) -> torch.Tensor:
        if tuple(state.shape) != self.shape:
            raise ValueError(f"State shape {tuple(state.shape)} does not match simulator shape {self.shape}")
        dims = tuple(range(state.ndim))
        state_hat = torch.fft.rfftn(state, dim=dims)
        return torch.fft.irfftn(state_hat * self.kernel_hat, s=self.shape, dim=dims)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(device=self.device, dtype=self.dtype)
        self._ensure_step_callable()
        assert self._step_callable is not None
        try:
            return self._step_callable(state, self.kernel_hat)
        except Exception as exception:
            if self.compile_step:
                raise RuntimeError(
                    "torch.compile Lenia step failed "
                    f"(backend={self.compile_backend}, mode={self.compile_mode}, device={self.device})."
                ) from exception
            raise

    def rollout(
        self,
        initial_state: torch.Tensor,
        steps: int,
        *,
        snapshot_interval: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        state = initial_state.to(device=self.device, dtype=self.dtype)
        snapshots: list[torch.Tensor] = []
        if snapshot_interval is not None and snapshot_interval > 0:
            snapshots.append(state)
        for step_index in range(1, int(steps) + 1):
            state = self.step(state)
            if snapshot_interval is not None and snapshot_interval > 0 and step_index % snapshot_interval == 0:
                snapshots.append(state)
        return state, snapshots


def make_seed_state(
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 1,
) -> torch.Tensor:
    """Create a compact deterministic soft blob field without voxel Python loops."""

    device = torch.device(device or "cpu")
    if device.type != "cuda":
        raise RuntimeError("make_seed_state is GPU-only; pass device='cuda'.")
    require_cuda()
    shape = tuple(int(v) for v in shape)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    axes = [torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype) for size in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    radius2 = torch.zeros(shape, device=device, dtype=dtype)
    for grid in grids:
        radius2 = radius2 + grid * grid
    blob = torch.exp(-radius2 / 0.12)
    noise = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    shell = torch.exp(-((torch.sqrt(radius2) - 0.45) ** 2) / 0.018)
    state = 0.32 * blob + 0.18 * shell + 0.08 * noise * (radius2 < 0.55)
    return torch.clamp(state, 0.0, 1.0)
