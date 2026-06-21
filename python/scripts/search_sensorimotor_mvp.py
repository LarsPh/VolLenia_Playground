#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only before uv sync installs PyYAML.
    yaml = None
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_diff.export_cpp import tensor_to_f32_file, write_catalog
from vollenia_diff.io import load_catalog_animal_state, resample_state
from vollenia_diff.metrics import descriptors_to_json, normalized_descriptors, state_summary
from vollenia_diff.params import LeniaParams
from vollenia_diff.rollout_losses import PROFILE_LOSSES, rollout_collect, terms_to_json
from vollenia_diff.simulator import LeniaSimulator, make_seed_state, require_cuda


class RatePairColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        unit = str(task.fields.get("unit", "it"))
        speed = task.speed
        if speed is None or speed <= 0.0:
            return Text(f"-- {unit}/s -- s/{unit}")
        return Text(f"{speed:.2f} {unit}/s {1.0 / speed:.2f} s/{unit}")


class StatusColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(str(task.fields.get("status", "")))


class SearchProgress:
    def __init__(self) -> None:
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            RatePairColumn(),
            StatusColumn(),
            console=self.console,
        )

    def __enter__(self) -> "SearchProgress":
        if self.progress is not None:
            self.progress.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.progress is not None:
            self.progress.stop()

    def add_task(self, description: str, *, total: int, unit: str, status: str = "") -> int | None:
        return self.progress.add_task(description, total=max(int(total), 0), unit=unit, status=status)

    def update(self, task_id: int | None, *, advance: int = 0, status: str | None = None) -> None:
        if task_id is None:
            return
        fields = {}
        if status is not None:
            fields["status"] = status
        self.progress.update(task_id, advance=advance, refresh=True, **fields)

    def remove(self, task_id: int | None) -> None:
        if task_id is None:
            return
        self.progress.refresh()
        self.progress.remove_task(task_id)

    def log(self, message: str) -> None:
        self.progress.console.print(message)


DEFAULT_ARGS: dict[str, Any] = {
    "catalog": Path("configs/lenia3d_reference/animals.json"),
    "animal": "",
    "size": 128,
    "source_size": 64,
    "steps": 50,
    "iterations": 160,
    "random_init_count": 4,
    "inner_optim_steps": 125,
    "train_clip_mode": "straight_through_hard",
    "eval_clip_mode": "hard",
    "optimize_params": "m,s",
    "seed": 0,
    "source_selection": "best",
    "mutation_std": None,
    "mutation": {
        "probability": 0.0,
        "no_mutation_every_iterations": 0,
        "mutated_inner_optim_steps": None,
        "initial_logit_probability": 1.0,
        "initial_logit_std": 0.0,
        "rule": {
            "enabled": False,
            "probability": 1.0,
            "R_probability": 1.0,
            "R_log_std": 0.0,
            "m_probability": 1.0,
            "m_std": 0.0,
            "s_probability": 1.0,
            "s_log_std": 0.0,
            "b_probability": 1.0,
            "b_element_probability": 1.0,
            "b_std": 0.0,
            "T_probability": 1.0,
            "T_log_std": 0.0,
        },
    },
    "rule_randomization": {
        "enabled": False,
        "probability": 1.0,
        "R_probability": 1.0,
        "R_log_std": 0.0,
        "m_probability": 1.0,
        "m_std": 0.0,
        "s_probability": 1.0,
        "s_log_std": 0.0,
        "b_probability": 1.0,
        "b_element_probability": 1.0,
        "b_std": 0.0,
        "T_probability": 1.0,
        "T_log_std": 0.0,
        "ring_count_probability": 0.0,
        "ring_count_choices": [],
        "kn_probability": 0.0,
        "kn_choices": [],
        "gn_probability": 0.0,
        "gn_choices": [],
    },
    "objective": {},
    "export_top_k": 3,
    "save_initial_states": True,
    "checkpoint_every_iterations": 5,
    "best_every_iterations": 1,
    "debug_allow_size_32": False,
    "compile_step": False,
    "optimizer_name": "Adam",
    "optimizer_lr": 0.08,
    "optimizer_initial_lr": None,
    "optimizer_param_lr": None,
    "max_grad_norm": None,
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    if text in {"null", "None", "~"}:
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")


def _load_simple_yaml_config(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        data = _load_simple_yaml_config(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Search config must be a mapping: {path}")
    return data


def _config_to_args(config: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    flat_keys = set(DEFAULT_ARGS) | {"profile", "out"}
    for key in flat_keys:
        if key in config:
            args[key] = config[key]

    source = config.get("source", {})
    if isinstance(source, dict):
        for source_key, arg_key in (
            ("catalog", "catalog"),
            ("animal", "animal"),
            ("source_size", "source_size"),
        ):
            if source_key in source:
                args[arg_key] = source[source_key]
        if isinstance(source.get("rule_randomization"), dict):
            args["rule_randomization"] = source["rule_randomization"]

    simulation = config.get("simulation", {})
    if isinstance(simulation, dict):
        for sim_key in ("size", "steps"):
            if sim_key in simulation:
                args[sim_key] = simulation[sim_key]

    search = config.get("search", {})
    if isinstance(search, dict):
        for search_key, arg_key in (
            ("iterations", "iterations"),
            ("random_init_count", "random_init_count"),
            ("source_selection", "source_selection"),
            ("mutation_std", "mutation_std"),
            ("debug_allow_size_32", "debug_allow_size_32"),
            ("seed", "seed"),
        ):
            if search_key in search:
                args[arg_key] = search[search_key]
        if isinstance(search.get("mutation"), dict):
            args["mutation"] = search["mutation"]

    optimization = config.get("optimization", {})
    if isinstance(optimization, dict):
        for opt_key, arg_key in (
            ("inner_optim_steps", "inner_optim_steps"),
            ("train_clip_mode", "train_clip_mode"),
            ("eval_clip_mode", "eval_clip_mode"),
            ("optimize_params", "optimize_params"),
            ("compile_step", "compile_step"),
        ):
            if opt_key in optimization:
                args[arg_key] = optimization[opt_key]
        optimizer = optimization.get("optimizer", {})
        if isinstance(optimizer, dict):
            for opt_key, arg_key in (
                ("name", "optimizer_name"),
                ("lr", "optimizer_lr"),
                ("initial_lr", "optimizer_initial_lr"),
                ("param_lr", "optimizer_param_lr"),
                ("max_grad_norm", "max_grad_norm"),
            ):
                if opt_key in optimizer:
                    args[arg_key] = optimizer[opt_key]

    export = config.get("export", {})
    if isinstance(export, dict):
        for export_key, arg_key in (
            ("out", "out"),
            ("export_top_k", "export_top_k"),
            ("save_initial_states", "save_initial_states"),
            ("checkpoint_every_iterations", "checkpoint_every_iterations"),
            ("best_every_iterations", "best_every_iterations"),
        ):
            if export_key in export:
                args[arg_key] = export[export_key]
    if isinstance(config.get("objective"), dict):
        args["objective"] = config["objective"]
    return args


def _path_or_none(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, Path) else Path(value)


def _coerce_float_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_float_fields(item) for key, item in value.items()}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    return value


def _coerce_args(values: dict[str, Any]) -> argparse.Namespace:
    result = dict(values)
    for key in ("catalog", "out"):
        if key in result:
            result[key] = _path_or_none(result[key])
    if isinstance(result.get("optimize_params"), list):
        result["optimize_params"] = ",".join(str(v) for v in result["optimize_params"])
    for key in (
        "size",
        "source_size",
        "steps",
        "iterations",
        "random_init_count",
        "inner_optim_steps",
        "export_top_k",
        "checkpoint_every_iterations",
        "best_every_iterations",
        "seed",
    ):
        if key in result and result[key] is not None:
            result[key] = int(result[key])
    for key in ("mutation_std", "optimizer_lr", "optimizer_initial_lr", "optimizer_param_lr", "max_grad_norm"):
        if key in result and result[key] is not None:
            result[key] = float(result[key])
    result["objective"] = copy.deepcopy(result.get("objective") or {})
    result["rule_randomization"] = _deep_update(DEFAULT_ARGS["rule_randomization"], result.get("rule_randomization") or {})
    result["mutation"] = _deep_update(DEFAULT_ARGS["mutation"], result.get("mutation") or {})
    if result.get("mutation_std") is not None:
        result["mutation"]["initial_logit_std"] = float(result["mutation_std"])
    result["mutation_std"] = float(result["mutation"].get("initial_logit_std", 0.0))
    result["rule_randomization"] = _coerce_float_fields(result["rule_randomization"])
    result["mutation"] = _coerce_float_fields(result["mutation"])
    for key in ("debug_allow_size_32", "compile_step", "save_initial_states"):
        result[key] = bool(result.get(key, False))
    return argparse.Namespace(**result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan 07 Sensorimotor-style VolLenia search MVP.")
    parser.add_argument("--config", type=Path, default=None, help="YAML search config. Explicit CLI flags override config values.")
    parser.add_argument("--profile", choices=("move_shape_target", "maintain_animal_profile", "rescue_unstable_animal"), default=argparse.SUPPRESS, help="Goal/loss profile used for train loss and hard-eval ranking.")
    parser.add_argument("--catalog", type=Path, default=argparse.SUPPRESS, help="Input animal catalog JSON when starting from a catalog source.")
    parser.add_argument("--animal", default=argparse.SUPPRESS, help="Animal selector for --catalog: empty means first animal; accepts index or substring of slug/code/name/cname.")
    parser.add_argument("--size", type=int, default=argparse.SUPPRESS, help="Cubic simulation resolution N for an N^3 search rollout.")
    parser.add_argument("--source-size", type=int, default=argparse.SUPPRESS, help="Canonical procedural source resolution before resampling to --size.")
    parser.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Lenia rollout time steps per train/eval trajectory.")
    parser.add_argument("--iterations", type=int, default=argparse.SUPPRESS, help="Outer search iterations, i.e. number of candidates optimized, hard-evaluated, ranked, and archived.")
    parser.add_argument("--random-init-count", type=int, default=argparse.SUPPRESS, help="Number of procedural seed states inserted into the initial source pool before outer search starts.")
    parser.add_argument("--inner-optim-steps", type=int, default=argparse.SUPPRESS, help="Gradient optimizer steps per candidate. Each step runs one differentiable rollout of --steps Lenia steps.")
    parser.add_argument("--train-clip-mode", choices=("hard", "straight_through_hard"), default=argparse.SUPPRESS, help="Clip/surrogate mode used during BPTT training.")
    parser.add_argument("--eval-clip-mode", choices=("hard",), default=argparse.SUPPRESS, help="Evaluation clip mode used for final rollout and ranking.")
    parser.add_argument("--optimize-params", default=argparse.SUPPRESS, help="Comma-separated continuous rule params to optimize along with initial logits.")
    parser.add_argument("--out", type=Path, default=argparse.SUPPRESS, help="Output run directory for archive, metrics, summary, exported catalog, cells, and snapshots.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Base RNG seed for procedural source generation and torch randomness.")
    parser.add_argument("--source-selection", choices=("best", "nearest_goal"), default=argparse.SUPPRESS, help="How to pick the archive source for the next candidate.")
    parser.add_argument("--mutation-std", type=float, default=argparse.SUPPRESS, help="Stddev of Gaussian noise added to initial-state logits before optimizing each candidate.")
    parser.add_argument("--export-top-k", type=int, default=argparse.SUPPRESS, help="Number of highest-ranked candidates exported into catalog.json and cells/*.f32.")
    parser.add_argument("--save-initial-states", action="store_true", default=argparse.SUPPRESS, help="Write optimized rollout initial states to initials/*.f32 for exported candidates.")
    parser.add_argument("--no-save-initial-states", dest="save_initial_states", action="store_false", default=argparse.SUPPRESS, help="Disable writing optimized rollout initial states.")
    parser.add_argument("--checkpoint-every-iterations", type=int, default=argparse.SUPPRESS, help="Write a top-K checkpoint catalog every N outer iterations; 0 disables.")
    parser.add_argument("--best-every-iterations", type=int, default=argparse.SUPPRESS, help="Update out/best/catalog.json at this interval when a new best appears; 0 disables.")
    parser.add_argument("--debug-allow-size-32", action="store_true", default=argparse.SUPPRESS, help="Permit --size 32 only for pytest or very short debugging; not a Plan 07 acceptance/search smoke size.")
    parser.add_argument("--compile-step", action="store_true", default=argparse.SUPPRESS, help="Use torch.compile for compatible single-step simulator calls; logging/export/kernel generation stay outside compile.")
    parser.add_argument("--optimizer-name", default=argparse.SUPPRESS, help="Torch optimizer class name. Default Adam.")
    parser.add_argument("--optimizer-lr", type=float, default=argparse.SUPPRESS, help="Default optimizer learning rate. Default 0.08 preserves current behavior.")
    parser.add_argument("--optimizer-initial-lr", type=float, default=argparse.SUPPRESS, help="Optional separate learning rate for initial-state logits.")
    parser.add_argument("--optimizer-param-lr", type=float, default=argparse.SUPPRESS, help="Optional separate learning rate for m/s/T raw params.")
    parser.add_argument("--max-grad-norm", type=float, default=argparse.SUPPRESS, help="Optional global gradient clipping threshold.")
    parsed = parser.parse_args()

    values = copy.deepcopy(DEFAULT_ARGS)
    if parsed.config is not None:
        values = _deep_update(values, _config_to_args(_load_yaml_config(parsed.config)))
    overrides = {key: value for key, value in vars(parsed).items() if key != "config"}
    values.update(overrides)
    args = _coerce_args(values)
    args.config = parsed.config
    if not getattr(args, "profile", ""):
        parser.error("--profile is required unless provided by --config")
    if getattr(args, "out", None) is None:
        parser.error("--out is required unless provided by --config export.out")
    return args


def _logit_scalar(value: float, low: float, high: float, device: torch.device) -> torch.Tensor:
    scaled = min(max((float(value) - low) / (high - low), 1.0e-4), 1.0 - 1.0e-4)
    return torch.logit(torch.tensor(scaled, device=device, dtype=torch.float32))


def _inv_softplus(value: float, min_value: float, device: torch.device) -> torch.Tensor:
    shifted = max(float(value) - min_value, 1.0e-4)
    return torch.log(torch.expm1(torch.tensor(shifted, device=device, dtype=torch.float32)))


def _parse_optimize_params(value: str) -> set[str]:
    requested = {part.strip() for part in value.split(",") if part.strip()}
    unsupported = requested - {"m", "s", "T"}
    if unsupported:
        raise ValueError(
            f"Unsupported differentiable params {sorted(unsupported)}; Plan 07 supports only m,s,T. "
            "R/b/kernel differentiability is deferred to a later milestone."
        )
    return requested


def params_from_raw(base: LeniaParams, raw: dict[str, torch.Tensor]) -> tuple[LeniaParams, dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    if "m" in raw:
        tensors["m"] = 0.5 * torch.sigmoid(raw["m"])
    if "s" in raw:
        tensors["s"] = F.softplus(raw["s"]) + 1.0e-4
    if "T" in raw:
        tensors["T"] = F.softplus(raw["T"]) + 1.0
    params = replace(
        base,
        m=float(tensors.get("m", torch.tensor(base.m)).detach().cpu()),
        s=float(tensors.get("s", torch.tensor(base.s)).detach().cpu()),
        T=float(tensors.get("T", torch.tensor(base.T)).detach().cpu()),
    )
    return params, tensors


def make_raw_params(base: LeniaParams, requested: set[str], device: torch.device) -> dict[str, torch.Tensor]:
    raw: dict[str, torch.Tensor] = {}
    if "m" in requested:
        raw["m"] = _logit_scalar(base.m, 0.0, 0.5, device).detach().requires_grad_(True)
    if "s" in requested:
        raw["s"] = _inv_softplus(base.s, 1.0e-4, device).detach().requires_grad_(True)
    if "T" in requested:
        raw["T"] = _inv_softplus(base.T, 1.0, device).detach().requires_grad_(True)
    return raw


def make_optimizer(args: argparse.Namespace, logits: torch.Tensor, raw_params: dict[str, torch.Tensor]) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, str(args.optimizer_name), None)
    if optimizer_class is None:
        raise ValueError(f"Unknown torch optimizer: {args.optimizer_name}")
    base_lr = float(args.optimizer_lr)
    initial_lr = args.optimizer_initial_lr
    param_lr = args.optimizer_param_lr
    if initial_lr is None and param_lr is None:
        return optimizer_class([logits, *raw_params.values()], lr=base_lr)
    groups: list[dict[str, Any]] = [{"params": [logits], "lr": float(initial_lr if initial_lr is not None else base_lr)}]
    if raw_params:
        groups.append({"params": list(raw_params.values()), "lr": float(param_lr if param_lr is not None else base_lr)})
    return optimizer_class(groups, lr=base_lr)


def _tensor_grad_norm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        return torch.zeros((), device=tensor.device)
    return tensor.grad.detach().norm()


def gradient_diagnostics(
    logits: torch.Tensor,
    raw_params: dict[str, torch.Tensor],
    *,
    max_grad_norm: float | None,
) -> dict[str, float]:
    params = [logits, *raw_params.values()]
    global_before = torch.sqrt(
        sum((_tensor_grad_norm(param) ** 2 for param in params if param.grad is not None), torch.zeros((), device=logits.device))
    )
    clipped_to: float | None = None
    if max_grad_norm is not None and max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        clipped_to = float(max_grad_norm)
    global_after = torch.sqrt(
        sum((_tensor_grad_norm(param) ** 2 for param in params if param.grad is not None), torch.zeros((), device=logits.device))
    )
    logit_norm = _tensor_grad_norm(logits)
    diagnostics = {
        "grad_norm_global": float(global_before.detach().cpu()),
        "grad_norm_global_after_clip": float(global_after.detach().cpu()),
        "logit_grad_norm": float(logit_norm.detach().cpu()),
        "logit_grad_rms": float((logit_norm / (logits.numel() ** 0.5)).detach().cpu()),
        "max_grad_norm": float(clipped_to) if clipped_to is not None else 0.0,
    }
    for name, tensor in raw_params.items():
        diagnostics[f"{name}_grad"] = float(_tensor_grad_norm(tensor).detach().cpu())
    for name in ("m", "s", "T"):
        diagnostics.setdefault(f"{name}_grad", 0.0)
    return diagnostics


def choose_source(archive_memory: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not archive_memory:
        raise ValueError("Archive is empty")
    for item in archive_memory:
        if item.get("score") == float("-inf") and not item.get("used", False):
            return item
    scored = [item for item in archive_memory if item.get("score") != float("-inf")]
    if not scored:
        return archive_memory[0]
    if mode == "nearest_goal":
        return max(scored, key=lambda item: item["score"])
    return max(scored, key=lambda item: item["score"])


def source_label(source: dict[str, Any]) -> str:
    metadata = source.get("source", source)
    if "seed" in metadata:
        return f"seed={metadata['seed']}"
    for key in ("slug", "code", "name", "kind"):
        value = metadata.get(key)
        if value:
            return f"{key}={value}"
    return "source=unknown"


def _scale_params_radius(params: LeniaParams, scale: float) -> LeniaParams:
    if scale == 1.0:
        return params
    return replace(params, R=float(params.R) * float(scale))


def _normal_scalar(std: float, generator: torch.Generator | None) -> float:
    if float(std) == 0.0:
        return 0.0
    return float(torch.randn((), generator=generator).item()) * float(std)


def _uniform_scalar(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator).item())


def _bernoulli(probability: float, generator: torch.Generator | None) -> bool:
    return _uniform_scalar(generator) < _clamp(float(probability), 0.0, 1.0)


def _choice(values: list[Any], generator: torch.Generator | None) -> Any:
    if not values:
        raise ValueError("Cannot sample from an empty choice list")
    index = int(torch.randint(len(values), (), generator=generator).item())
    return values[index]


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _field_probability(config: dict[str, Any], name: str) -> float:
    return float(config.get(f"{name}_probability", 1.0))


def _apply_rule_noise(
    params: LeniaParams,
    config: dict[str, Any] | None,
    *,
    seed: int | None,
    label: str,
    allow_structure: bool = False,
) -> tuple[LeniaParams, dict[str, Any]]:
    config = config or {}
    before = params.to_catalog_dict()
    if not bool(config.get("enabled", False)):
        return params, {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "label": label,
            "before": before,
            "after": before,
            "fields": {},
        }
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    if not _bernoulli(float(config.get("probability", 1.0)), generator):
        return params, {
            "enabled": True,
            "applied": False,
            "reason": "probability",
            "label": label,
            "seed": seed,
            "config": copy.deepcopy(config),
            "before": before,
            "after": before,
            "fields": {},
        }

    R = float(params.R)
    m = float(params.m)
    s = float(params.s)
    T = float(params.T)
    b = [float(value) for value in params.b]
    kn = int(params.kn)
    gn = int(params.gn)
    fields: dict[str, Any] = {}

    if allow_structure and _bernoulli(float(config.get("ring_count_probability", 0.0)), generator):
        choices = [max(int(value), 1) for value in _as_list(config.get("ring_count_choices"))]
        if choices:
            ring_count = int(_choice(choices, generator))
            old_b = b
            b = [old_b[index] if index < len(old_b) else _uniform_scalar(generator) for index in range(ring_count)]
            fields["ring_count"] = {"applied": True, "before": len(old_b), "after": ring_count}
    if allow_structure and _bernoulli(float(config.get("kn_probability", 0.0)), generator):
        choices = [int(value) for value in _as_list(config.get("kn_choices"))]
        if choices:
            old_kn = kn
            kn = int(_choice(choices, generator))
            fields["kn"] = {"applied": True, "before": old_kn, "after": kn}
    if allow_structure and _bernoulli(float(config.get("gn_probability", 0.0)), generator):
        choices = [int(value) for value in _as_list(config.get("gn_choices"))]
        if choices:
            old_gn = gn
            gn = int(_choice(choices, generator))
            fields["gn"] = {"applied": True, "before": old_gn, "after": gn}

    if _bernoulli(_field_probability(config, "R"), generator):
        old = R
        R = R * float(torch.exp(torch.tensor(_normal_scalar(float(config.get("R_log_std", 0.0)), generator))).item())
        fields["R"] = {"applied": R != old, "before": old, "after": R}
    if _bernoulli(_field_probability(config, "m"), generator):
        old = m
        m = _clamp(m + _normal_scalar(float(config.get("m_std", 0.0)), generator), 1.0e-4, 0.499)
        fields["m"] = {"applied": m != old, "before": old, "after": m}
    if _bernoulli(_field_probability(config, "s"), generator):
        old = s
        s = s * float(torch.exp(torch.tensor(_normal_scalar(float(config.get("s_log_std", 0.0)), generator))).item())
        fields["s"] = {"applied": s != old, "before": old, "after": s}
    if _bernoulli(_field_probability(config, "T"), generator):
        old = T
        T = T * float(torch.exp(torch.tensor(_normal_scalar(float(config.get("T_log_std", 0.0)), generator))).item())
        fields["T"] = {"applied": T != old, "before": old, "after": T}
    b_applied = _bernoulli(_field_probability(config, "b"), generator)
    b_std = float(config.get("b_std", 0.0))
    old_b = list(b)
    b_mask: list[bool] = []
    if b_applied:
        element_probability = float(config.get("b_element_probability", 1.0))
        for index, value in enumerate(b):
            element_applied = _bernoulli(element_probability, generator)
            b_mask.append(element_applied)
            if element_applied:
                b[index] = _clamp(float(value) + _normal_scalar(b_std, generator), 0.0, 1.0)
    fields["b"] = {"applied": b != old_b, "before": old_b, "after": b, "mask": b_mask}
    noisy = replace(params, R=max(R, 1.0e-5), m=m, s=max(s, 1.0e-5), T=max(T, 1.0), b=b, kn=kn, gn=gn).sanitized()
    after = noisy.to_catalog_dict()
    changed = after != before
    metadata = {
        "enabled": True,
        "applied": changed,
        "reason": "applied" if changed else "no_field_changed",
        "label": label,
        "seed": seed,
        "config": copy.deepcopy(config),
        "before": before,
        "after": after,
        "fields": fields,
    }
    return noisy, metadata


def _initial_logit_std(mutation: dict[str, Any]) -> float:
    return float(mutation.get("initial_logit_std", 0.0) or 0.0)


def candidate_mutation_decision(
    mutation: dict[str, Any],
    *,
    iteration: int,
    seed: int,
    full_inner_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    full_inner_steps = max(int(full_inner_steps), 1)
    disabled = copy.deepcopy(mutation)
    disabled["initial_logit_std"] = 0.0
    disabled["rule"] = {**dict(disabled.get("rule", {})), "enabled": False}

    no_mutation_every = int(float(mutation.get("no_mutation_every_iterations") or 0))
    if no_mutation_every > 0 and (int(iteration) + 1) % no_mutation_every == 0:
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "scheduled_no_mutation",
            "iteration": int(iteration),
            "seed": int(seed),
            "inner_steps_used": full_inner_steps,
            "config": copy.deepcopy(mutation),
        }
        return disabled, metadata, full_inner_steps

    if not _bernoulli(float(mutation.get("probability", 1.0)), generator):
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "mutation_probability",
            "iteration": int(iteration),
            "seed": int(seed),
            "inner_steps_used": full_inner_steps,
            "config": copy.deepcopy(mutation),
        }
        return disabled, metadata, full_inner_steps

    effective = copy.deepcopy(mutation)
    initial_std = _initial_logit_std(effective)
    initial_applied = initial_std > 0.0 and _bernoulli(float(effective.get("initial_logit_probability", 1.0)), generator)
    if not initial_applied:
        effective["initial_logit_std"] = 0.0

    rule_config = dict(effective.get("rule", {}))
    rule_applied = bool(rule_config.get("enabled", False)) and _bernoulli(float(rule_config.get("probability", 1.0)), generator)
    if not rule_applied:
        rule_config["enabled"] = False
    effective["rule"] = rule_config

    applied = bool(initial_applied or rule_applied)
    if not applied:
        metadata = {
            "enabled": True,
            "applied": False,
            "reason": "no_component_selected",
            "iteration": int(iteration),
            "seed": int(seed),
            "inner_steps_used": full_inner_steps,
            "initial_logit": {"applied": False, "std": 0.0},
            "rule": {"applied": False},
            "config": copy.deepcopy(mutation),
        }
        return disabled, metadata, full_inner_steps

    mutated_inner_steps = effective.get("mutated_inner_optim_steps")
    inner_steps_used = full_inner_steps if mutated_inner_steps is None else min(max(int(float(mutated_inner_steps)), 1), full_inner_steps)
    metadata = {
        "enabled": True,
        "applied": True,
        "reason": "mutation",
        "iteration": int(iteration),
        "seed": int(seed),
        "inner_steps_used": inner_steps_used,
        "initial_logit": {"applied": bool(initial_applied), "std": float(effective.get("initial_logit_std", 0.0) or 0.0)},
        "rule": {"applied": bool(rule_applied)},
        "config": copy.deepcopy(mutation),
    }
    return effective, metadata, inner_steps_used


def procedural_source_state(
    size: int,
    source_size: int,
    seed: int,
    device: torch.device,
    rule_randomization: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, LeniaParams, dict[str, Any]]:
    source_shape = (int(source_size), int(source_size), int(source_size))
    state = make_seed_state(source_shape, device=device, seed=seed)
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
        "rule_randomization": rule_metadata,
    }
    return state, params, metadata


def source_from_args(args: argparse.Namespace, device: torch.device, *, seed: int | None = None) -> tuple[torch.Tensor, LeniaParams, dict[str, Any]]:
    if args.profile in {"maintain_animal_profile", "rescue_unstable_animal"} or args.animal:
        source = load_catalog_animal_state(args.catalog, args.animal or None, size=args.size, device=device)
        params, rule_metadata = _apply_rule_noise(
            source.params,
            args.rule_randomization,
            seed=seed,
            label="source_randomization",
            allow_structure=True,
        )
        return source.state, params, {"kind": "catalog", **source.source_metadata, "rule_randomization": rule_metadata}
    return procedural_source_state(args.size, args.source_size, int(args.seed if seed is None else seed), device, args.rule_randomization)


def initial_archive_sources(args: argparse.Namespace, device: torch.device, progress: SearchProgress) -> list[dict[str, Any]]:
    source_state, source_params, source_metadata = source_from_args(args, device, seed=int(args.seed))
    sources = [{"state0": source_state.detach(), "params": source_params, "score": float("-inf"), "source": source_metadata, "used": False}]
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
            state, params, metadata = procedural_source_state(args.size, args.source_size, seed, device, args.rule_randomization)
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


def target_for_profile(profile: str, initial: torch.Tensor, objective: dict[str, Any] | None = None) -> torch.Tensor | None:
    if profile != "move_shape_target":
        return None
    objective = objective or {}
    offset_norm = objective.get("target_offset_norm_zyx", [0.0, 0.0, 0.12])
    com = normalized_descriptors(initial)["com_norm"] * torch.tensor(
        [max(size - 1, 1) for size in initial.shape],
        device=initial.device,
        dtype=initial.dtype,
    )
    offset = torch.tensor([float(value) * float(min(initial.shape)) for value in offset_norm], device=initial.device, dtype=initial.dtype)
    max_coord = torch.tensor([size - 1 for size in initial.shape], device=initial.device, dtype=initial.dtype)
    return torch.minimum(torch.maximum(com + offset, torch.zeros_like(com)), max_coord)


def score_100_from_loss(loss_value: float) -> float:
    return 100.0 / (1.0 + max(float(loss_value), 0.0))


def _range_violation(value: float, low: float, high: float, low_reason: str, high_reason: str) -> str | None:
    if value < low:
        return low_reason
    if value > high:
        return high_reason
    return None


def evaluate_life_gate(descriptor: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    gates = objective.get("eval_gates", {}) if isinstance(objective, dict) else {}
    if not isinstance(gates, dict) or not gates:
        return {"life_gate_pass": True, "collapse_reason": "ok", "gate_violations": [], "eval_gates": {}}

    violations: list[dict[str, Any]] = []

    def add(reason: str, name: str, value: float, threshold: Any) -> None:
        violations.append({"reason": reason, "metric": name, "value": value, "threshold": threshold})

    max_density = float(descriptor.get("max_density", 0.0))
    min_max_density = gates.get("min_max_density")
    if min_max_density is not None and max_density < float(min_max_density):
        add("dead_density", "max_density", max_density, float(min_max_density))

    mass_fraction = float(descriptor.get("mass_fraction", 0.0))
    mass_window = gates.get("mass_fraction")
    if mass_window is not None:
        low, high = [float(value) for value in mass_window]
        reason = _range_violation(mass_fraction, low, high, "dead_mass", "over_mass_noise")
        if reason:
            add(reason, "mass_fraction", mass_fraction, [low, high])

    active_fraction = float(descriptor.get("active_fraction", 0.0))
    active_window = gates.get("active_fraction")
    if active_window is not None:
        low, high = [float(value) for value in active_window]
        reason = _range_violation(active_fraction, low, high, "under_active", "over_active_noise")
        if reason:
            add(reason, "active_fraction", active_fraction, [low, high])

    body_radius_norm = float(descriptor.get("body_radius_norm", 0.0))
    body_radius_window = gates.get("body_radius_norm")
    if body_radius_window is not None:
        low, high = [float(value) for value in body_radius_window]
        reason = _range_violation(body_radius_norm, low, high, "too_compact", "too_diffuse")
        if reason:
            add(reason, "body_radius_norm", body_radius_norm, [low, high])

    border_mass = float(descriptor.get("border_mass", 0.0))
    max_border_mass = gates.get("max_border_mass")
    if max_border_mass is not None and border_mass > float(max_border_mass):
        add("border_leak", "border_mass", border_mass, float(max_border_mass))

    reason_priority = [
        "dead_density",
        "dead_mass",
        "under_active",
        "over_active_noise",
        "over_mass_noise",
        "too_diffuse",
        "border_leak",
        "too_compact",
    ]
    reasons = [item["reason"] for item in violations]
    collapse_reason = "ok"
    for reason in reason_priority:
        if reason in reasons:
            collapse_reason = reason
            break
    return {
        "life_gate_pass": not violations,
        "collapse_reason": collapse_reason,
        "gate_violations": violations,
        "eval_gates": copy.deepcopy(gates),
    }


def evaluate_candidate(
    state0: torch.Tensor,
    params: LeniaParams,
    *,
    profile: str,
    target: torch.Tensor | None,
    steps: int,
    clip_mode: str,
    compile_step: bool,
    objective: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, float], float]:
    simulator = LeniaSimulator(state0.shape, params, device=state0.device, clip_mode=clip_mode, compile_step=compile_step)
    states = rollout_collect(simulator, state0, steps, params=params, clip_mode=clip_mode, sample_interval=max(steps // 4, 1))
    loss_fn = PROFILE_LOSSES[profile]
    if target is None:
        loss, terms = loss_fn(states, objective=objective)
    else:
        loss, terms = loss_fn(states, target=target, objective=objective)
    final = states[-1]
    descriptor_tensor = normalized_descriptors(final, initial=state0, target=target)
    descriptor = descriptors_to_json(descriptor_tensor)
    summary = state_summary(final, target)
    loss_value = float(loss.detach().cpu())
    loss_score = -loss_value
    score_100 = score_100_from_loss(loss_value)
    gate = evaluate_life_gate({**summary, **descriptor}, objective)
    gate_fail_score_cap = float(objective.get("gate_fail_score_cap", 5.0)) if isinstance(objective, dict) else 5.0
    rank_score_100 = score_100 if gate["life_gate_pass"] else min(score_100, gate_fail_score_cap)
    score = rank_score_100
    terms_json = terms_to_json(terms)
    terms_json["loss_score"] = loss_score
    terms_json["score_100"] = score_100
    terms_json["rank_score_100"] = rank_score_100
    return final, {**summary, "descriptor": descriptor, **gate, "score_100": score_100, "rank_score_100": rank_score_100, "loss_score": loss_score}, terms_json, score


def optimize_candidate(
    source_state0: torch.Tensor,
    base_params: LeniaParams,
    *,
    args: argparse.Namespace,
    profile: str,
    target: torch.Tensor | None,
    steps: int,
    inner_steps: int,
    train_clip_mode: str,
    optimize_params: set[str],
    mutation_std: float,
    mutation: dict[str, Any],
    mutation_decision: dict[str, Any],
    mutation_seed: int,
    compile_step: bool,
    progress: SearchProgress,
    progress_task: int | None,
) -> tuple[torch.Tensor, LeniaParams, dict[str, Any], dict[str, float], list[dict[str, float]], dict[str, float]]:
    base_params, rule_mutation_metadata = _apply_rule_noise(
        base_params,
        dict(mutation.get("rule", {})),
        seed=mutation_seed,
        label="candidate_mutation",
    )
    initial = torch.clamp(source_state0.detach(), 1.0e-4, 1.0 - 1.0e-4)
    logits = torch.logit(initial).detach().requires_grad_(True)
    if mutation_std > 0.0:
        with torch.no_grad():
            generator = torch.Generator(device=logits.device).manual_seed(int(mutation_seed) + 17)
            logits.add_(torch.randn(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator) * mutation_std)
    raw_params = make_raw_params(base_params, optimize_params, source_state0.device)
    optimizer = make_optimizer(args, logits, raw_params)
    simulator = LeniaSimulator(source_state0.shape, base_params, device=source_state0.device, clip_mode=train_clip_mode, compile_step=compile_step)
    grad_history: list[dict[str, float]] = []
    last_grad_diagnostics: dict[str, float] = {
        "grad_norm_global": 0.0,
        "grad_norm_global_after_clip": 0.0,
        "logit_grad_norm": 0.0,
        "logit_grad_rms": 0.0,
        "m_grad": 0.0,
        "s_grad": 0.0,
        "T_grad": 0.0,
        "max_grad_norm": 0.0,
    }
    last_terms: dict[str, float] = {}
    final_train = source_state0
    params = base_params

    for step_index in range(int(inner_steps)):
        optimizer.zero_grad(set_to_none=True)
        state0 = torch.sigmoid(logits)
        params, tensor_params = params_from_raw(base_params, raw_params)
        states = rollout_collect(
            simulator,
            state0,
            steps,
            params=base_params,
            m=tensor_params.get("m"),
            s=tensor_params.get("s"),
            T=tensor_params.get("T"),
            clip_mode=train_clip_mode,
            sample_interval=max(steps // 4, 1),
        )
        loss_fn = PROFILE_LOSSES[profile]
        if target is None:
            loss, terms = loss_fn(states, objective=args.objective)
        else:
            loss, terms = loss_fn(states, target=target, objective=args.objective)
        loss.backward()
        last_grad_diagnostics = gradient_diagnostics(logits, raw_params, max_grad_norm=args.max_grad_norm)
        grad_history.append(last_grad_diagnostics)
        optimizer.step()
        final_train = states[-1].detach()
        last_terms = terms_to_json(terms)
        term_status = " ".join(
            f"{key}={last_terms[key]:.3g}"
            for key in ("com", "target_mask", "mass_ratio", "active_ratio", "compactness_ratio", "visibility")
            if key in last_terms
        )
        progress.update(
            progress_task,
            advance=1,
            status=(
                f"step={step_index + 1} loss={last_terms.get('loss_total', 0.0):.4g} "
                f"logit_rms={last_grad_diagnostics['logit_grad_rms']:.3g} "
                f"m={last_grad_diagnostics['m_grad']:.3g} s={last_grad_diagnostics['s_grad']:.3g} "
                f"T={last_grad_diagnostics['T_grad']:.3g} {term_status}"
            ),
        )

    learned_state0 = torch.sigmoid(logits).detach()
    params, _ = params_from_raw(base_params, raw_params)
    final_summary = state_summary(final_train.detach(), target)
    final_summary["candidate_mutation"] = {
        **copy.deepcopy(mutation_decision),
        "initial_logit_std": float(mutation_std),
        "rule": rule_mutation_metadata,
    }
    return learned_state0, params, final_summary, last_terms, grad_history, last_grad_diagnostics


def ranked_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(entries, key=lambda item: item.get("rank_score_100", item.get("score", float("-inf"))), reverse=True)
    for rank, entry in enumerate(ranked):
        entry["rank"] = rank
    return ranked


def write_catalog_for_entries(
    out_dir: Path,
    entries: list[dict[str, Any]],
    export_states: dict[str, torch.Tensor],
    initial_states: dict[str, torch.Tensor],
    export_params: dict[str, LeniaParams],
    *,
    args: argparse.Namespace,
    metrics: dict[str, Any] | None = None,
) -> Path:
    states = {entry["id"]: export_states[entry["id"]] for entry in entries}
    params = {entry["id"]: export_params[entry["id"]] for entry in entries}
    metadata = {
        entry["id"]: {
            "score": entry["score"],
            "score_100": entry.get("score_100", entry["score"]),
            "rank_score_100": entry.get("rank_score_100", entry["score"]),
            "loss_score": entry.get("loss_score", 0.0),
            "loss_total": entry.get("loss_total", 0.0),
            "life_gate_pass": entry.get("life_gate_pass", True),
            "collapse_reason": entry.get("collapse_reason", "ok"),
            "gate_violations": entry.get("gate_violations", []),
            "rank": entry.get("rank", -1),
            "goal_profile": entry["goal_profile"],
            "objective_terms": entry["objective_terms"],
            "objective_config": entry.get("objective_config", {}),
            "train_clip_mode": entry["train_clip_mode"],
            "eval_clip_mode": entry["eval_clip_mode"],
            "export_activation": "raw",
            "source": entry["source"],
            "candidate_mutation": entry.get("metrics_train", {}).get("candidate_mutation", {}),
            "inner_steps_used": entry.get("inner_steps_used", 0),
            "optimizer": entry.get("optimizer", {}),
            "gradient_diagnostics": entry.get("gradient_diagnostics", {}),
            "learned_initial_file": f"initials/{entry['id']}_initial.f32" if args.save_initial_states and entry["id"] in initial_states else "",
        }
        for entry in entries
    }
    initial_subset: dict[str, torch.Tensor] = {}
    if args.save_initial_states:
        initials_dir = out_dir / "initials"
        for entry in entries:
            state = initial_states.get(entry["id"])
            if state is not None:
                initial_subset[entry["id"]] = state
                tensor_to_f32_file(state, initials_dir / f"{entry['id']}_initial.f32")
    manifest_path = write_catalog(
        states,
        out_dir,
        params,
        source="vollenia_search_mvp",
        metrics=metrics,
        simulation_dims={key: [args.size, args.size, args.size] for key in states},
        resolution_policy={key: "native" for key in states},
        animal_metadata=metadata,
    )
    if args.save_initial_states and initial_subset:
        initial_metadata = {
            entry["id"]: {
                "score": entry["score"],
                "score_100": entry.get("score_100", entry["score"]),
                "rank_score_100": entry.get("rank_score_100", entry["score"]),
                "life_gate_pass": entry.get("life_gate_pass", True),
                "collapse_reason": entry.get("collapse_reason", "ok"),
                "rank": entry.get("rank", -1),
                "goal_profile": entry["goal_profile"],
                "state_role": "learned_initial",
                "final_cells_file": f"../cells/{entry['id']}.f32",
                "source": entry["source"],
                "optimizer": entry.get("optimizer", {}),
            }
            for entry in entries
            if entry["id"] in initial_subset
        }
        write_catalog(
            initial_subset,
            out_dir / "initial_catalog",
            params,
            source="vollenia_search_mvp_initial_states",
            metrics=metrics,
            simulation_dims={key: [args.size, args.size, args.size] for key in initial_subset},
            resolution_policy={key: "native" for key in initial_subset},
            animal_metadata=initial_metadata,
        )
    return manifest_path


def maybe_write_periodic_catalogs(
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    export_states: dict[str, torch.Tensor],
    initial_states: dict[str, torch.Tensor],
    export_params: dict[str, LeniaParams],
    *,
    iteration: int,
    is_new_best: bool,
) -> None:
    if not entries:
        return
    ranked = ranked_entries(entries)
    checkpoint_every = int(args.checkpoint_every_iterations)
    if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
        top = ranked[: max(int(args.export_top_k), 1)]
        checkpoint_dir = args.out / "checkpoints" / f"iter_{iteration + 1:05d}"
        write_catalog_for_entries(checkpoint_dir, top, export_states, initial_states, export_params, args=args, metrics={"entries": top})
    best_every = int(args.best_every_iterations)
    if is_new_best and best_every > 0 and ((iteration + 1) % best_every == 0 or iteration == 0):
        write_catalog_for_entries(args.out / "best", ranked[:1], export_states, initial_states, export_params, args=args, metrics={"entries": ranked[:1]})


def write_outputs(
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    export_states: dict[str, torch.Tensor],
    initial_states: dict[str, torch.Tensor],
    export_params: dict[str, LeniaParams],
    run_started: float,
    progress: SearchProgress,
) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    export_task = progress.add_task("Export outputs", total=6, unit="file", status="archive")
    ranked = ranked_entries(entries)
    (args.out / "archive.json").write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
    progress.update(export_task, advance=1, status="metrics")
    (args.out / "metrics.json").write_text(json.dumps({"entries": ranked}, indent=2) + "\n", encoding="utf-8")
    progress.update(export_task, advance=1, status="candidates.csv")

    with (args.out / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "rank",
                "score",
                "score_100",
                "rank_score_100",
                "loss_total",
                "loss_score",
                "life_gate_pass",
                "collapse_reason",
                "goal_profile",
                "source_slug",
                "train_clip_mode",
                "eval_clip_mode",
            ],
        )
        writer.writeheader()
        for entry in ranked:
            writer.writerow({
                "id": entry["id"],
                "rank": entry["rank"],
                "score": entry["score"],
                "score_100": entry.get("score_100", entry["score"]),
                "rank_score_100": entry.get("rank_score_100", entry["score"]),
                "loss_total": entry.get("loss_total", 0.0),
                "loss_score": entry.get("loss_score", 0.0),
                "life_gate_pass": entry.get("life_gate_pass", True),
                "collapse_reason": entry.get("collapse_reason", "ok"),
                "goal_profile": entry["goal_profile"],
                "source_slug": entry["source"].get("slug", entry["source"].get("kind", "")),
                "train_clip_mode": entry["train_clip_mode"],
                "eval_clip_mode": entry["eval_clip_mode"],
            })
    progress.update(export_task, advance=1, status="catalog")

    top = ranked[: max(int(args.export_top_k), 1)]
    write_catalog_for_entries(args.out, top, export_states, initial_states, export_params, args=args, metrics=None)
    progress.update(export_task, advance=1, status="snapshots")
    snapshots_dir = args.out / "snapshots"
    for entry in top:
        tensor_to_f32_file(export_states[entry["id"]], snapshots_dir / f"{entry['id']}_final.f32")
    progress.update(export_task, advance=1, status="summary")

    elapsed = time.perf_counter() - run_started
    summary = [
        f"# Sensorimotor Search MVP Summary",
        "",
        f"- profile: `{args.profile}`",
        f"- size: `{args.size}`",
        f"- steps: `{args.steps}`",
        f"- iterations: `{args.iterations}`",
        f"- inner_optim_steps: `{args.inner_optim_steps}`",
        f"- train_clip_mode: `{args.train_clip_mode}`",
        f"- eval_clip_mode: `{args.eval_clip_mode}`",
        f"- optimize_params: `{args.optimize_params or 'none'}`",
        f"- objective: `{json.dumps(args.objective, sort_keys=True)}`",
        f"- mutation: `{json.dumps(args.mutation, sort_keys=True)}`",
        f"- rule_randomization: `{json.dumps(args.rule_randomization, sort_keys=True)}`",
        f"- source_size: `{args.source_size}`",
        f"- optimizer: `{args.optimizer_name}(lr={args.optimizer_lr}, initial_lr={args.optimizer_initial_lr}, param_lr={args.optimizer_param_lr}, max_grad_norm={args.max_grad_norm})`",
        f"- checkpoint_every_iterations: `{args.checkpoint_every_iterations}`",
        f"- best_every_iterations: `{args.best_every_iterations}`",
        f"- save_initial_states: `{args.save_initial_states}`",
        f"- config: `{args.config if args.config is not None else 'none'}`",
        f"- device: `{torch.cuda.get_device_name(0)}`",
        f"- elapsed_seconds: `{elapsed:.3f}`",
        f"- best_score: `{top[0]['score'] if top else 'n/a'}`",
        f"- best_rank_score_100: `{top[0].get('rank_score_100', 'n/a') if top else 'n/a'}`",
        f"- best_life_gate_pass: `{top[0].get('life_gate_pass', 'n/a') if top else 'n/a'}`",
        f"- best_collapse_reason: `{top[0].get('collapse_reason', 'n/a') if top else 'n/a'}`",
        "",
        "Catalog can be opened from the C++ Animal Catalog file picker.",
    ]
    if args.debug_allow_size_32 and args.size == 32:
        summary.append("\nDebug-only run: size 32 is not a Plan 07 search acceptance size.")
    (args.out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    progress.update(export_task, advance=1, status="done")
    progress.remove(export_task)


def main() -> None:
    require_cuda()
    args = parse_args()
    if args.size < 64 and not args.debug_allow_size_32:
        raise ValueError("Plan 07 search runs require --size >= 64; use --debug-allow-size-32 only for tests.")
    optimize_params = _parse_optimize_params(args.optimize_params)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    run_started = time.perf_counter()

    with SearchProgress() as progress:
        archive_memory = initial_archive_sources(args, device, progress)
        target = target_for_profile(args.profile, archive_memory[0]["state0"], args.objective)
        entries: list[dict[str, Any]] = []
        export_states: dict[str, torch.Tensor] = {}
        initial_states: dict[str, torch.Tensor] = {}
        export_params: dict[str, LeniaParams] = {}
        search_task = progress.add_task("Search candidates", total=args.iterations, unit="cand", status="starting")
        best_score = float("-inf")

        for iteration in range(int(args.iterations)):
            iter_start = time.perf_counter()
            source = choose_source(archive_memory, args.source_selection)
            source["used"] = True
            current_source = source_label(source)
            mutation_seed = int(args.seed) + iteration * 1000003
            effective_mutation, mutation_decision, inner_steps_used = candidate_mutation_decision(
                args.mutation,
                iteration=iteration,
                seed=mutation_seed,
                full_inner_steps=args.inner_optim_steps,
            )
            mutation_status = "mutate" if mutation_decision.get("applied") else str(mutation_decision.get("reason", "no_mutation"))
            progress.update(search_task, status=f"iter={iteration} {mutation_status} {current_source}")
            inner_task = progress.add_task(
                f"Inner BPTT {iteration:03d}",
                total=inner_steps_used,
                unit="step",
                status=f"{mutation_status} {current_source}",
            )
            learned_state0, learned_params, metrics_train, terms_train, grad_history, last_grad_diagnostics = optimize_candidate(
                source["state0"],
                source["params"],
                args=args,
                profile=args.profile,
                target=target,
                steps=args.steps,
                inner_steps=inner_steps_used,
                train_clip_mode=args.train_clip_mode,
                optimize_params=optimize_params,
                mutation_std=float(effective_mutation.get("initial_logit_std", 0.0) or 0.0),
                mutation=effective_mutation,
                mutation_decision=mutation_decision,
                mutation_seed=mutation_seed,
                compile_step=args.compile_step,
                progress=progress,
                progress_task=inner_task,
            )
            progress.remove(inner_task)
            eval_task = progress.add_task(f"Hard eval {iteration:03d}", total=1, unit="eval", status="rollout")
            final_eval, metrics_eval, terms_eval, score = evaluate_candidate(
                learned_state0,
                learned_params,
                profile=args.profile,
                target=target,
                steps=args.steps,
                clip_mode=args.eval_clip_mode,
                compile_step=args.compile_step,
                objective=args.objective,
            )
            progress.update(eval_task, advance=1, status=f"score={score:.4g}")
            progress.remove(eval_task)
            candidate_id = f"{args.profile}_{iteration:04d}"
            candidate_source = source.get("source", {})
            optimizer_config = {
                "name": args.optimizer_name,
                "lr": args.optimizer_lr,
                "initial_lr": args.optimizer_initial_lr,
                "param_lr": args.optimizer_param_lr,
                "max_grad_norm": args.max_grad_norm,
            }
            entry = {
                "id": candidate_id,
                "source": candidate_source,
                "goal_profile": args.profile,
                "params": learned_params.to_catalog_dict(),
                "train_clip_mode": args.train_clip_mode,
                "eval_clip_mode": args.eval_clip_mode,
                "metrics_train": metrics_train,
                "metrics_eval": metrics_eval,
                "descriptor": metrics_eval.get("descriptor", {}),
                "objective_terms": {"train": terms_train, "eval": terms_eval},
                "objective_config": copy.deepcopy(args.objective),
                "gradient_history": grad_history,
                "gradient_diagnostics": last_grad_diagnostics,
                "optimizer": optimizer_config,
                "mutation_config": copy.deepcopy(args.mutation),
                "mutation_decision": copy.deepcopy(metrics_train.get("candidate_mutation", mutation_decision)),
                "inner_steps_used": inner_steps_used,
                "score": score,
                "score_100": metrics_eval.get("score_100", score),
                "rank_score_100": metrics_eval.get("rank_score_100", score),
                "loss_score": metrics_eval.get("loss_score", terms_eval.get("loss_score", -terms_eval.get("loss_total", 0.0))),
                "loss_total": terms_eval.get("loss_total", 0.0),
                "life_gate_pass": metrics_eval.get("life_gate_pass", True),
                "collapse_reason": metrics_eval.get("collapse_reason", "ok"),
                "gate_violations": metrics_eval.get("gate_violations", []),
                "rank": -1,
                "artifact_paths": {
                    "cells": f"cells/{candidate_id}.f32",
                    "initial": f"initials/{candidate_id}_initial.f32" if args.save_initial_states else "",
                    "snapshot": f"snapshots/{candidate_id}_final.f32",
                },
                "timing_seconds": time.perf_counter() - iter_start,
            }
            entries.append(entry)
            export_states[candidate_id] = final_eval.detach()
            initial_states[candidate_id] = learned_state0.detach()
            export_params[candidate_id] = learned_params
            archive_memory.append({"state0": learned_state0.detach(), "params": learned_params, "score": score, "source": candidate_source})
            is_new_best = score > best_score
            best_score = max(best_score, score)
            maybe_write_periodic_catalogs(
                args,
                entries,
                export_states,
                initial_states,
                export_params,
                iteration=iteration,
                is_new_best=is_new_best,
            )
            progress.update(search_task, advance=1, status=f"best={best_score:.4g} last={score:.4g}")
            main_terms = terms_eval if terms_eval else terms_train
            term_status = " ".join(
                f"{key}={main_terms[key]:.3g}"
                for key in ("com", "balanced_target", "absolute_occupancy", "mass_ratio", "active_ratio", "compactness_ratio", "visibility")
                if key in main_terms
            )
            progress.log(
                f"iter={iteration:03d} score_100={metrics_eval.get('score_100', score):.6f} "
                f"rank_score_100={metrics_eval.get('rank_score_100', score):.6f} "
                f"life_gate={metrics_eval.get('life_gate_pass', True)} "
                f"collapse={metrics_eval.get('collapse_reason', 'ok')} "
                f"logit_grad_rms={last_grad_diagnostics['logit_grad_rms']:.6g} "
                f"m_grad={last_grad_diagnostics['m_grad']:.6g} "
                f"s_grad={last_grad_diagnostics['s_grad']:.6g} "
                f"T_grad={last_grad_diagnostics['T_grad']:.6g} "
                f"loss={terms_eval.get('loss_total', terms_train.get('loss_total', 0.0)):.6g} {term_status} "
                f"mutation={metrics_train.get('candidate_mutation', {}).get('reason', mutation_status)} "
                f"inner_steps={inner_steps_used} "
                f"max_density={metrics_eval['max_density']:.4f} active={metrics_eval['active_voxels']} "
                f"time={entry['timing_seconds']:.3f}s"
            )

        write_outputs(args, entries, export_states, initial_states, export_params, run_started, progress)
        progress.log(f"Wrote search outputs to {args.out}")


if __name__ == "__main__":
    main()
