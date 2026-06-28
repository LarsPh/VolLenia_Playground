#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_model.spec import load_model_spec
from vollenia_model.visualization import plot_model_kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ModelSpec smooth/legacy kernels.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_model_spec(args.spec)
    paths = plot_model_kernels(spec, args.out, device=args.device)
    print(f"Wrote {len(paths)} kernel plot files to {args.out}")


if __name__ == "__main__":
    main()
