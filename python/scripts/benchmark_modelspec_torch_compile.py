#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_model.expanded_flow import ModelSpecSimulator, seed_state
from vollenia_model.parity import benchmark_step
from vollenia_model.spec import load_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager vs torch.compile ModelSpec step.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_model_spec(args.spec)
    dims = (args.size, args.size, args.size)
    sim = ModelSpecSimulator(spec, dims=dims, device=args.device)
    state = seed_state(spec, dims=dims, device=args.device)
    try:
        result = benchmark_step(sim.step, state, repeats=args.steps, compile_step=not args.no_compile)
        payload = {
            "spec": str(args.spec),
            "size": args.size,
            "steps": args.steps,
            "eager_ms": result.eager_ms,
            "compiled_ms": result.compiled_ms,
            "max_abs_diff": result.max_abs_diff,
            "compile_error": None,
        }
    except Exception as exc:
        result = benchmark_step(sim.step, state, repeats=args.steps, compile_step=False)
        payload = {
            "spec": str(args.spec),
            "size": args.size,
            "steps": args.steps,
            "eager_ms": result.eager_ms,
            "compiled_ms": None,
            "max_abs_diff": None,
            "compile_error": str(exc),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "benchmark.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
