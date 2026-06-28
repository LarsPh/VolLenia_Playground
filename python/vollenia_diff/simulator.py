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
    if clip_mode == "straight_through_hard":
        return 2
    if clip_mode == "none":
        return 0
    raise ValueError(f"Unknown clip_mode: {clip_mode}")


def _scalar_like(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _growth_from_id(u: torch.Tensor, m: float | torch.Tensor, s: float | torch.Tensor, gn: int) -> torch.Tensor:
    mu = _scalar_like(m, u)
    sigma = torch.clamp(_scalar_like(s, u), min=1.0e-5)
    diff = u - mu
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
    m: float | torch.Tensor,
    s: float | torch.Tensor,
    T: float | torch.Tensor,
    gn: int,
    clip_mode_id: int,
) -> torch.Tensor:
    """Compile-friendly single Lenia step. Keep logging/export outside this function."""

    dims = tuple(range(state.ndim))
    state_hat = torch.fft.rfftn(state, dim=dims)
    u = torch.fft.irfftn(state_hat * kernel_hat, s=shape, dim=dims)
    next_state = state + _growth_from_id(u, m, s, gn) / torch.clamp(_scalar_like(T, state), min=1.0)
    if clip_mode_id == 1:
        return torch.clamp(next_state, 0.0, 1.0)
    if clip_mode_id == 2:
        clipped = torch.clamp(next_state, 0.0, 1.0)
        return next_state + (clipped - next_state).detach()
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
        self._dynamic_step_callable = None
        self._dynamic_step_key: tuple[object, ...] | None = None
        self.dynamic_compile_error: str | None = None

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
        self._dynamic_step_callable = None
        self._dynamic_step_key = None

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

    def _current_dynamic_step_key(self) -> tuple[object, ...]:
        params = self.params.sanitized()
        return (
            self.shape,
            str(self.device),
            str(self.dtype),
            params.gn,
            self.clip_mode,
            self.compile_step,
            self.compile_backend,
            self.compile_mode,
        )

    def _ensure_dynamic_step_callable(self) -> None:
        self._ensure_kernel()
        key = self._current_dynamic_step_key()
        if key == self._dynamic_step_key and self._dynamic_step_callable is not None:
            return
        params = self.params.sanitized()
        clip_id = _clip_mode_id(self.clip_mode)

        def step_fn(
            state: torch.Tensor,
            kernel_hat: torch.Tensor,
            m: torch.Tensor,
            s: torch.Tensor,
            T: torch.Tensor,
        ) -> torch.Tensor:
            return lenia_step_tensor(state, kernel_hat, self.shape, m, s, T, params.gn, clip_id)

        if self.compile_step:
            self._dynamic_step_callable = torch.compile(step_fn, backend=self.compile_backend, mode=self.compile_mode)
        else:
            self._dynamic_step_callable = step_fn
        self._dynamic_step_key = key

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

    def dynamic_step(
        self,
        state: torch.Tensor,
        *,
        m: float | torch.Tensor,
        s: float | torch.Tensor,
        T: float | torch.Tensor,
    ) -> torch.Tensor:
        state = state.to(device=self.device, dtype=self.dtype)
        self._ensure_dynamic_step_callable()
        assert self._dynamic_step_callable is not None
        m_tensor = _scalar_like(m, state)
        s_tensor = _scalar_like(s, state)
        T_tensor = _scalar_like(T, state)
        try:
            return self._dynamic_step_callable(state, self.kernel_hat, m_tensor, s_tensor, T_tensor)
        except Exception as exception:
            if self.compile_step:
                self.dynamic_compile_error = (
                    "torch.compile dynamic Lenia step failed; fell back to eager "
                    f"(backend={self.compile_backend}, mode={self.compile_mode}, device={self.device}): {exception}"
                )
                params = self.params.sanitized()
                return lenia_step_tensor(
                    state,
                    self.kernel_hat,
                    self.shape,
                    m_tensor,
                    s_tensor,
                    T_tensor,
                    params.gn,
                    _clip_mode_id(self.clip_mode),
                )
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
    mode: str = "blob_shell",
    config: dict[str, object] | None = None,
) -> torch.Tensor:
    """Create a deterministic procedural seed field without voxel Python loops."""

    device = torch.device(device or "cpu")
    if device.type != "cuda":
        raise RuntimeError("make_seed_state is GPU-only; pass device='cuda'.")
    require_cuda()
    shape = tuple(int(v) for v in shape)
    config = config or {}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    axes = [torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype) for size in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    radius2 = torch.zeros(shape, device=device, dtype=dtype)
    for grid in grids:
        radius2 = radius2 + grid * grid

    if mode == "random_patch":
        min_dim = min(shape)
        size_range = config.get("patch_size_fraction", [0.28, 0.55])
        low_frac, high_frac = [float(value) for value in size_range]  # type: ignore[arg-type]
        low_size = max(2, int(round(low_frac * min_dim)))
        high_size = max(low_size, int(round(high_frac * min_dim)))
        patch_size = int(torch.randint(low_size, high_size + 1, (), device=device, generator=generator).item())
        patch_shape = tuple(min(patch_size, size) for size in shape)
        patch = torch.rand(patch_shape, device=device, dtype=dtype, generator=generator)
        density_scale = float(config.get("patch_density_scale", 0.65))
        density_bias = float(config.get("patch_density_bias", 0.0))
        state = torch.zeros(shape, device=device, dtype=dtype)
        slices = tuple(slice((size - patch) // 2, (size - patch) // 2 + patch) for size, patch in zip(shape, patch_shape, strict=True))
        state[slices] = torch.clamp(density_bias + density_scale * patch, 0.0, 1.0)
        return state

    if mode == "mixed_blobs":
        count_range = config.get("blob_count", [2, 5])
        low_count, high_count = [int(value) for value in count_range]  # type: ignore[arg-type]
        blob_count = int(torch.randint(max(low_count, 1), max(high_count, low_count) + 1, (), device=device, generator=generator).item())
        state = torch.zeros(shape, device=device, dtype=dtype)
        for _ in range(blob_count):
            center = [
                float(torch.empty((), device=device, dtype=dtype).uniform_(-0.45, 0.45, generator=generator).item())
                for _axis in shape
            ]
            radius = float(torch.empty((), device=device, dtype=dtype).uniform_(0.12, 0.38, generator=generator).item())
            amplitude = float(torch.empty((), device=device, dtype=dtype).uniform_(0.10, 0.55, generator=generator).item())
            blob_radius2 = torch.zeros(shape, device=device, dtype=dtype)
            for grid, center_value in zip(grids, center, strict=True):
                blob_radius2 = blob_radius2 + (grid - center_value) * (grid - center_value)
            state = state + amplitude * torch.exp(-blob_radius2 / max(radius * radius, 1.0e-4))
        shell_probability = float(config.get("shell_probability", 0.35))
        if float(torch.rand((), device=device, generator=generator).item()) < shell_probability:
            shell_radius = float(torch.empty((), device=device, dtype=dtype).uniform_(0.25, 0.60, generator=generator).item())
            shell_width = float(torch.empty((), device=device, dtype=dtype).uniform_(0.010, 0.045, generator=generator).item())
            shell_amp = float(torch.empty((), device=device, dtype=dtype).uniform_(0.05, 0.25, generator=generator).item())
            state = state + shell_amp * torch.exp(-((torch.sqrt(radius2) - shell_radius) ** 2) / shell_width)
        noise_amp = float(config.get("noise_amplitude", 0.04))
        if noise_amp > 0.0:
            state = state + noise_amp * torch.rand(shape, device=device, dtype=dtype, generator=generator) * (radius2 < 0.75)
        return torch.clamp(state, 0.0, 1.0)

    if mode != "blob_shell":
        raise ValueError(f"Unknown procedural seed mode: {mode}")

    blob = torch.exp(-radius2 / 0.12)
    noise = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    shell = torch.exp(-((torch.sqrt(radius2) - 0.45) ** 2) / 0.018)
    state = 0.32 * blob + 0.18 * shell + 0.08 * noise * (radius2 < 0.55)
    return torch.clamp(state, 0.0, 1.0)
