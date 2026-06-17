from __future__ import annotations

import torch

from vollenia_diff.metrics import center_of_mass
from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import LeniaSimulator, make_seed_state


def test_initial_state_logits_receive_nonzero_gradient() -> None:
    shape = (12, 12, 12)
    simulator = LeniaSimulator(shape, LeniaParams(R=4.0), device="cuda", clip_mode="none")
    initial = make_seed_state(shape, device="cuda", seed=3)
    logits = torch.logit(torch.clamp(initial, 1.0e-4, 1.0 - 1.0e-4)).detach().requires_grad_(True)
    final_state, _ = simulator.rollout(torch.sigmoid(logits), 3)
    target = torch.tensor([6.0, 6.0, 7.0], device="cuda")
    loss = torch.linalg.vector_norm(center_of_mass(final_state) - target) ** 2
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm() > 0
