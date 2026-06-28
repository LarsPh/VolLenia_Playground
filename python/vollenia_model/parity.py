from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class BenchmarkResult:
    eager_ms: float
    compiled_ms: float | None
    max_abs_diff: float | None


def benchmark_step(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    *,
    repeats: int = 8,
    compile_step: bool = True,
) -> BenchmarkResult:
    def run(fn: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, float]:
        value = state
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            value = fn(value)
        torch.cuda.synchronize()
        return value, (time.perf_counter() - start) * 1000.0 / max(1, repeats)

    eager_value, eager_ms = run(step_fn)
    if not compile_step:
        return BenchmarkResult(eager_ms=eager_ms, compiled_ms=None, max_abs_diff=None)
    compiled_fn = torch.compile(step_fn)
    compiled_value, compiled_ms = run(compiled_fn)
    diff = torch.max(torch.abs(eager_value - compiled_value)).item()
    return BenchmarkResult(eager_ms=eager_ms, compiled_ms=compiled_ms, max_abs_diff=diff)
