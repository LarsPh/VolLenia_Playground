#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import LeniaSimulator, lenia_step_tensor, make_seed_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager vs torch.compile Lenia CUDA steps.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--out", type=Path, default=Path("outputs/diff_bridge/compile_benchmark"))
    parser.add_argument("--print-graph", action="store_true", help="Print the torch._dynamo exported FX graph.")
    return parser.parse_args()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for this benchmark.")


def exported_graph_code(shape: tuple[int, int, int], state: torch.Tensor, kernel_hat: torch.Tensor) -> str:
    def step_fn(state_arg: torch.Tensor, kernel_hat_arg: torch.Tensor) -> torch.Tensor:
        return lenia_step_tensor(state_arg, kernel_hat_arg, shape, 0.12, 0.01, 10.0, 1, 1)

    exported = torch._dynamo.export(step_fn)(state, kernel_hat)
    return exported.graph_module.code


def warmup(simulator: LeniaSimulator, initial: torch.Tensor, steps: int) -> None:
    state = initial.clone()
    for _ in range(steps):
        state = simulator.step(state)
    torch.cuda.synchronize()


def time_rollout(simulator: LeniaSimulator, initial: torch.Tensor, steps: int) -> tuple[float, torch.Tensor]:
    state = initial.clone()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        state = simulator.step(state)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), state


def summarize(times_ms: list[float], steps: int) -> dict[str, float]:
    per_step = [value / float(steps) for value in times_ms]
    return {
        "total_ms_mean": statistics.fmean(times_ms),
        "total_ms_min": min(times_ms),
        "total_ms_max": max(times_ms),
        "per_step_ms_mean": statistics.fmean(per_step),
        "per_step_ms_min": min(per_step),
        "per_step_ms_max": max(per_step),
        "per_step_ms_std": statistics.pstdev(per_step) if len(per_step) > 1 else 0.0,
    }


def benchmark_one(simulator: LeniaSimulator, initial: torch.Tensor, args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor]:
    warmup(simulator, initial, args.warmup_steps)
    times: list[float] = []
    final = initial
    for _ in range(args.repeats):
        elapsed_ms, final = time_rollout(simulator, initial, args.steps)
        times.append(elapsed_ms)
    summary = summarize(times, args.steps)
    summary["runs_total_ms"] = times
    return summary, final


def main() -> None:
    require_cuda()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    shape = (args.size, args.size, args.size)
    params = LeniaParams()
    initial = make_seed_state(shape, device="cuda", seed=args.seed)
    eager = LeniaSimulator(shape, params, device="cuda")
    compiled = LeniaSimulator(
        shape,
        params,
        device="cuda",
        compile_step=True,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )

    graph_code = exported_graph_code(shape, initial, eager.kernel_hat)
    complex_warning_note = (
        "The Inductor complex-operator warning is caused by the FFT frequency-domain path: "
        "`torch._C._fft.fft_rfftn`, complex `state_hat * kernel_hat`, and "
        "`torch._C._fft.fft_irfftn`. The exported graph's explicit complex multiply node is "
        "`mul = state_hat * kernel_hat_`."
    )
    if args.print_graph:
        print(graph_code)
        print(complex_warning_note)

    torch.cuda.synchronize()
    compile_start = time.perf_counter()
    compiled_first = compiled.step(initial)
    torch.cuda.synchronize()
    compile_first_step_wall_ms = (time.perf_counter() - compile_start) * 1000.0

    eager_summary, eager_final = benchmark_one(eager, initial, args)
    compiled_summary, compiled_final = benchmark_one(compiled, initial, args)
    max_abs_diff = float((eager_final - compiled_final).abs().max().detach().cpu())
    compiled_first_diff = float((eager.step(initial) - compiled_first).abs().max().detach().cpu())

    result: dict[str, Any] = {
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "shape": list(shape),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "compile_first_step_wall_ms": compile_first_step_wall_ms,
        "eager": eager_summary,
        "compiled": compiled_summary,
        "compiled_vs_eager_max_abs_diff": max_abs_diff,
        "compiled_first_step_vs_eager_max_abs_diff": compiled_first_diff,
        "complex_warning_note": complex_warning_note,
        "exported_graph_code": graph_code,
    }

    output_path = args.out / "benchmark.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "exported_graph_code"}, indent=2))
    print(f"Wrote benchmark JSON to {output_path}")


if __name__ == "__main__":
    main()
