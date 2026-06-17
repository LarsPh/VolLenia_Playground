#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_diff.export_cpp import tensor_to_f32_file, write_catalog
from vollenia_diff.metrics import rollout_summary
from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import LeniaSimulator, make_seed_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small differentiable PyTorch Lenia rollout.")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--snapshot-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preset", default="diguttome_saliens", help="Metadata label only for Plan 06.")
    parser.add_argument("--out", type=Path, default=Path("outputs/diff_bridge/smoke_rollout"))
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--compile-step", action="store_true", help="Use torch.compile for the cached Lenia step.")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    return parser.parse_args()


def pick_device(value: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the VolLenia differentiable backend.")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    shape = (args.size, args.size, args.size) if args.dim == 3 else (args.size, args.size)
    params = LeniaParams()
    simulator = LeniaSimulator(
        shape,
        params,
        device=device,
        compile_step=args.compile_step,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )
    initial = make_seed_state(shape, device=device, seed=args.seed)
    final, snapshots = simulator.rollout(initial, args.steps, snapshot_interval=args.snapshot_interval)

    states: dict[str, torch.Tensor] = {"step_0000": initial.detach()}
    snapshot_dir = args.out / "snapshots"
    tensor_to_f32_file(initial, snapshot_dir / "step_0000.f32")
    for snapshot_index, snapshot in enumerate(snapshots[1:], start=1):
        step = snapshot_index * args.snapshot_interval
        slug = f"step_{step:04d}"
        states[slug] = snapshot.detach()
        tensor_to_f32_file(snapshot, snapshot_dir / f"{slug}.f32")
    if f"step_{args.steps:04d}" not in states:
        states[f"step_{args.steps:04d}"] = final.detach()
        tensor_to_f32_file(final, snapshot_dir / f"step_{args.steps:04d}.f32")

    metrics = rollout_summary(list(states.values()))
    metrics["device"] = str(device)
    metrics["compile_step"] = args.compile_step
    metrics["preset"] = args.preset
    metrics["steps"] = args.steps
    write_catalog(states, args.out, params, metrics=metrics)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote PyTorch rollout bridge catalog to {args.out / 'catalog.json'}")


if __name__ == "__main__":
    main()
