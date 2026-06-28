from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from typing import Any

import torch

from vollenia_diff.io import load_catalog_animal_state, resample_state
from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import make_seed_state

from .mutation import _apply_rule_noise, _scale_params_radius

def source_label(source: dict[str, Any]) -> str:
    metadata = source.get("source", source)
    if "seed" in metadata:
        return f"seed={metadata['seed']}"
    for key in ("slug", "code", "name", "kind"):
        value = metadata.get(key)
        if value:
            return f"{key}={value}"
    return "source=unknown"

def procedural_source_state(
    size: int,
    source_size: int,
    seed: int,
    device: torch.device,
    rule_randomization: dict[str, Any] | None = None,
    initialization: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, LeniaParams, dict[str, Any]]:
    source_shape = (int(source_size), int(source_size), int(source_size))
    initialization = copy.deepcopy(initialization or {})
    init_mode = str(initialization.get("mode", "blob_shell"))
    state = make_seed_state(source_shape, device=device, seed=seed, mode=init_mode, config=initialization)
    target_shape = (int(size), int(size), int(size))
    if source_shape != target_shape:
        state = resample_state(state, target_shape)
    scale = float(size) / float(source_size)
    native_params, rule_metadata = _apply_rule_noise(
        LeniaParams(),
        rule_randomization,
        seed=seed,
        label="source_randomization",
        allow_structure=True,
    )
    params = _scale_params_radius(native_params, scale)
    metadata = {
        "kind": "procedural",
        "seed": seed,
        "dims": [int(size), int(size), int(size)],
        "source_size": int(source_size),
        "native_simulation_dims": [int(source_size), int(source_size), int(source_size)],
        "target_dims": [int(size), int(size), int(size)],
        "target_size": int(size),
        "rule_radius_scale": scale,
        "initialization": {
            "mode": init_mode,
            "config": initialization,
        },
        "rule_randomization": rule_metadata,
    }
    return state, params, metadata

def source_from_args(args: argparse.Namespace, device: torch.device, *, seed: int | None = None) -> tuple[torch.Tensor, LeniaParams, dict[str, Any]]:
    if args.profile == "rescue_unstable_animal" or args.animal:
        source = load_catalog_animal_state(args.catalog, args.animal or None, size=args.size, device=device)
        params, rule_metadata = _apply_rule_noise(
            source.params,
            args.rule_randomization,
            seed=seed,
            label="source_randomization",
            allow_structure=True,
        )
        return source.state, params, {"kind": "catalog", **source.source_metadata, "rule_randomization": rule_metadata}
    return procedural_source_state(
        args.size,
        args.source_size,
        int(args.seed if seed is None else seed),
        device,
        args.rule_randomization,
        args.source_initialization,
    )

def should_inject_source(args: argparse.Namespace, *, iteration: int, injections_done: int) -> bool:
    config = args.source_injection
    if not bool(config.get("enabled", False)):
        return False
    max_injections = int(float(config.get("max_injections") or 0))
    if max_injections > 0 and int(injections_done) >= max_injections:
        return False
    if bool(config.get("start_after_initial_pool", True)) and int(iteration) < int(args.random_init_count):
        return False
    every = max(int(float(config.get("every_iterations") or 0)), 0)
    if every <= 0:
        return False
    base_iteration = int(args.random_init_count) if bool(config.get("start_after_initial_pool", True)) else 0
    return (int(iteration) - base_iteration) >= 0 and (int(iteration) - base_iteration) % every == 0

def inject_procedural_source(
    args: argparse.Namespace,
    archive_memory: list[dict[str, Any]],
    *,
    device: torch.device,
    iteration: int,
    injections_done: int,
) -> dict[str, Any]:
    seed = int(args.seed) + 10_000_000 + int(injections_done)
    state, params, metadata = procedural_source_state(
        args.size,
        args.source_size,
        seed,
        device,
        args.rule_randomization,
        args.source_initialization,
    )
    metadata = {
        **metadata,
        "source_injection": {
            "enabled": True,
            "iteration": int(iteration),
            "injection_index": int(injections_done),
            "seed": seed,
            "config": copy.deepcopy(args.source_injection),
        },
    }
    item = {
        "entry_id": f"injected_source_{injections_done:04d}",
        "state0": state.detach(),
        "params": params,
        "score": float("-inf"),
        "source": metadata,
        "used": False,
        "fresh_injected": True,
    }
    archive_memory.append(item)
    return item

def initial_archive_sources(args: argparse.Namespace, device: torch.device, progress: SearchProgress) -> list[dict[str, Any]]:
    source_state, source_params, source_metadata = source_from_args(args, device, seed=int(args.seed))
    sources = [{"entry_id": "initial_source_0000", "state0": source_state.detach(), "params": source_params, "score": float("-inf"), "source": source_metadata, "used": False}]
    if source_metadata.get("kind") != "procedural" and not bool(args.rule_randomization.get("enabled", False)):
        return sources
    init_task = None
    if int(args.random_init_count) > 1:
        init_task = progress.add_task("Initialize sources", total=args.random_init_count, unit="seed", status=f"seed={args.seed}")
        progress.update(init_task, advance=1)
    for offset in range(1, max(int(args.random_init_count), 1)):
        seed = int(args.seed) + offset
        progress.update(init_task, status=f"seed={seed}")
        if source_metadata.get("kind") == "procedural":
            state, params, metadata = procedural_source_state(args.size, args.source_size, seed, device, args.rule_randomization, args.source_initialization)
        else:
            params, rule_metadata = _apply_rule_noise(
                source_params,
                args.rule_randomization,
                seed=seed,
                label="source_randomization",
                allow_structure=True,
            )
            state = source_state
            metadata = {**source_metadata, "seed": seed, "rule_randomization": rule_metadata}
        sources.append(
            {
                "entry_id": f"initial_source_{offset:04d}",
                "state0": state.detach(),
                "params": params,
                "score": float("-inf"),
                "source": metadata,
                "used": False,
            }
        )
        progress.update(init_task, advance=1)
    progress.remove(init_task)
    return sources

__all__ = [
    "initial_archive_sources",
    "inject_procedural_source",
    "procedural_source_state",
    "should_inject_source",
    "source_from_args",
    "source_label",
]
