#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_diff.export_cpp import load_tensor, write_single_state_catalog
from vollenia_diff.params import LeniaParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a torch tensor to a C++ Lenia animal catalog.")
    parser.add_argument("input", type=Path, help=".pt file containing a tensor or {'state': tensor}.")
    parser.add_argument("--out", type=Path, default=Path("outputs/diff_bridge/exported"))
    parser.add_argument("--slug", default="torch_export")
    parser.add_argument("--R", type=float, default=10.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=float, default=0.12)
    parser.add_argument("--s", type=float, default=0.01)
    parser.add_argument("--b", default="1.0,0.75,0.5833333,0.9166667")
    parser.add_argument("--kn", type=int, default=1)
    parser.add_argument("--gn", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = LeniaParams(
        R=args.R,
        T=args.T,
        m=args.m,
        s=args.s,
        b=[float(part.strip()) for part in args.b.split(",") if part.strip()],
        kn=args.kn,
        gn=args.gn,
    )
    state = load_tensor(args.input)
    path = write_single_state_catalog(state, args.out, params, slug=args.slug)
    print(path)


if __name__ == "__main__":
    main()
