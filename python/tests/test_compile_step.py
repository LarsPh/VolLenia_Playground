from __future__ import annotations

import torch

from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import LeniaSimulator, make_seed_state


def test_compiled_step_matches_eager_cuda() -> None:
    shape = (12, 12, 12)
    params = LeniaParams(R=4.0)
    state = make_seed_state(shape, device="cuda", seed=11)
    eager = LeniaSimulator(shape, params, device="cuda")
    compiled = LeniaSimulator(shape, params, device="cuda", compile_step=True)
    eager_out = eager.step(state)
    compiled_out = compiled.step(state)
    assert compiled_out.shape == eager_out.shape
    assert torch.isfinite(compiled_out).all()
    assert torch.allclose(compiled_out, eager_out, atol=1.0e-5, rtol=1.0e-5)


def test_compiled_dynamic_step_matches_eager_cuda() -> None:
    shape = (12, 12, 12)
    params = LeniaParams(R=4.0)
    state = make_seed_state(shape, device="cuda", seed=12)
    eager = LeniaSimulator(shape, params, device="cuda", clip_mode="straight_through_hard")
    compiled = LeniaSimulator(shape, params, device="cuda", clip_mode="straight_through_hard", compile_step=True)
    m = torch.tensor(params.m, device="cuda")
    s = torch.tensor(params.s, device="cuda")
    T = torch.tensor(params.T, device="cuda")
    eager_out = eager.dynamic_step(state, m=m, s=s, T=T)
    compiled_out = compiled.dynamic_step(state, m=m, s=s, T=T)
    assert compiled_out.shape == eager_out.shape
    assert torch.isfinite(compiled_out).all()
    assert torch.allclose(compiled_out, eager_out, atol=1.0e-5, rtol=1.0e-5)
