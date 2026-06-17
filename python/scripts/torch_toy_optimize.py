#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_diff.export_cpp import apply_export_activation, write_single_state_catalog
from vollenia_diff.metrics import center_of_mass, state_summary
from vollenia_diff.objectives import toy_loss
from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import LeniaSimulator, make_seed_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny differentiable Lenia optimization sanity check.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=Path("outputs/diff_bridge/toy"))
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--compile-step", action="store_true", help="Use torch.compile for the cached Lenia step.")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--export-activation",
        default="sigmoid",
        choices=("clamp", "sigmoid", "raw"),
        help=(
            "Map the final optimization state before C++ .f32 export. "
            "Default sigmoid makes unconstrained clip_mode=none states renderer-visible."
        ),
    )
    return parser.parse_args()


def pick_device(value: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the VolLenia differentiable backend.")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    shape = (args.size, args.size, args.size)
    params = LeniaParams()
    simulator = LeniaSimulator(
        shape,
        params,
        device=device,
        clip_mode="none",
        compile_step=args.compile_step,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )
    initial = make_seed_state(shape, device=device, seed=args.seed)
    logits = torch.logit(torch.clamp(initial, 1.0e-4, 1.0 - 1.0e-4)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    target = torch.tensor(
        [args.size * 0.5, args.size * 0.5, args.size * 0.62],
        device=device,
        dtype=torch.float32,
    )

    previous_loss: float | None = None
    final_state = torch.sigmoid(logits).detach()
    for iteration in range(args.iters):
        optimizer.zero_grad(set_to_none=True)
        state0 = torch.sigmoid(logits)
        final_state, _ = simulator.rollout(state0, args.steps)
        loss = toy_loss(
            final_state,
            target,
            mass_min=float(args.size**3) * 0.01,
            mass_max=float(args.size**3) * 0.35,
            max_second_moment=float(args.size**2) * 0.18,
            lambda_mass=1.0e-6,
            lambda_compact=1.0e-4,
            lambda_border=0.5,
        )
        loss.backward()
        grad_norm = logits.grad.detach().norm()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        grad_value = float(grad_norm.detach().cpu())
        trend = "" if previous_loss is None else f" delta={loss_value - previous_loss:+.6f}"
        previous_loss = loss_value
        com = center_of_mass(final_state.detach())
        print(
            f"iter={iteration:03d} loss={loss_value:.6f} grad_norm={grad_value:.6f} "
            f"com={[round(float(v), 3) for v in com.detach().cpu().tolist()]}{trend}"
        )

    raw_final = final_state.detach()
    export_state = apply_export_activation(raw_final, args.export_activation)
    metrics = {
        "export_activation": args.export_activation,
        "raw_final": state_summary(raw_final, target.detach()),
        "exported_final": state_summary(export_state, target.detach()),
    }
    write_single_state_catalog(export_state, args.out, params, slug=f"toy_final_{args.steps:04d}", target=target)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    exported = metrics["exported_final"]
    print(
        "export "
        f"activation={args.export_activation} "
        f"mass={exported['mass']:.6f} "
        f"max_density={exported['max_density']:.6f} "
        f"active_voxels={exported['active_voxels']}"
    )
    print(f"Wrote optimized PyTorch bridge catalog to {args.out / 'catalog.json'}")


if __name__ == "__main__":
    main()
