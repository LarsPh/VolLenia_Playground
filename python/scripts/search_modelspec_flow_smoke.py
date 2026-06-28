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
    parser = argparse.ArgumentParser(description="Minimal BPTT smoke over a ModelSpec model.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_model_spec(args.spec)
    dims = (args.size, args.size, args.size)
    sim = ModelSpecSimulator(spec, dims=dims, device=args.device)
    initial = seed_state(spec, dims=dims, device=args.device)
    logits = torch.logit(torch.clamp(initial, 1.0e-4, 1.0 - 1.0e-4)).detach().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=args.lr)
    best_loss = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    target_mass = initial.sum().detach()
    for i in range(args.iters):
        opt.zero_grad(set_to_none=True)
        state0 = torch.sigmoid(logits)
        final = rollout(sim, state0, args.steps)
        mass_loss = ((final.sum() - target_mass) / torch.clamp_min(target_mass, 1.0)).square()
        compact_loss = final.mean()
        loss = mass_loss + 0.01 * compact_loss
        loss.backward()
        opt.step()
        value = float(loss.detach().item())
        history.append({"iter": float(i), "loss": value, "mass": float(final.detach().sum().item())})
        if value < best_loss:
            best_loss = value
            best_state = final.detach()
    assert best_state is not None
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = export_model_state(best_state, args.out, spec, model_spec_path=args.spec, composite=True)
    summary = {"best_loss": best_loss, "history": history, "manifest": str(manifest)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
