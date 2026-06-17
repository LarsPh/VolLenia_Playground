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
