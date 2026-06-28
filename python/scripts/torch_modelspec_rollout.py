#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_model.expanded_flow import ModelSpecSimulator, rollout, seed_state
from vollenia_model.export_state import export_model_state
from vollenia_model.spec import load_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short PyTorch ModelSpec rollout and export ModelState.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA PyTorch is required")
    spec = load_model_spec(args.spec)
    dims = (args.size, args.size, args.size) if args.size is not None else spec.dims
    sim = ModelSpecSimulator(spec, dims=dims, device=args.device)
    state0 = seed_state(spec, dims=dims, device=args.device, seed=args.seed)
    state = rollout(sim, state0, args.steps)
    manifest = export_model_state(state, args.out, spec, model_spec_path=args.spec, composite=True)
    summary = {
        "spec": str(args.spec),
        "steps": args.steps,
        "dims": list(dims),
        "channels": spec.channel_count,
        "min": float(state.min().item()),
        "max": float(state.max().item()),
        "mass": float(state.sum().item()),
        "manifest": str(manifest),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
