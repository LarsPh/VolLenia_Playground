from __future__ import annotations

import torch

from vollenia_diff.kernels import build_radial_kernel
from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import LeniaSimulator, make_seed_state


def test_kernel_shape_and_normalization_2d() -> None:
    kernel = build_radial_kernel((24, 24), LeniaParams(R=6.0), device="cuda")
    assert kernel.shape == (24, 24)
    assert torch.isfinite(kernel).all()
    assert torch.allclose(kernel.sum(), torch.tensor(1.0, device="cuda"), atol=1.0e-5)


def test_kernel_shape_and_normalization_3d() -> None:
    kernel = build_radial_kernel((16, 16, 16), LeniaParams(R=5.0, b=[1.0, 0.5], kn=2), device="cuda")
    assert kernel.shape == (16, 16, 16)
    assert torch.isfinite(kernel).all()
    assert torch.allclose(kernel.sum(), torch.tensor(1.0, device="cuda"), atol=1.0e-5)


def test_step_shape_and_finite_values() -> None:
    shape = (16, 16, 16)
    simulator = LeniaSimulator(shape, LeniaParams(R=5.0), device="cuda")
    state = make_seed_state(shape, device="cuda", seed=7)
    next_state = simulator.step(state)
    assert next_state.shape == state.shape
    assert torch.isfinite(next_state).all()
    assert next_state.min() >= 0.0
    assert next_state.max() <= 1.0
