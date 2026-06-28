from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only before uv sync installs PyYAML.
    yaml = None

from vollenia_diff.rollout_losses import PROFILE_LOSSES

from .optimize import _parse_optimize_params

DEFAULT_ARGS: dict[str, Any] = {
    "catalog": Path("configs/lenia3d_reference/animals.json"),
    "animal": "",
    "size": 128,
    "source_size": 64,
    "source_initialization": {
        "mode": "blob_shell",
        "patch_size_fraction": [0.28, 0.55],
        "patch_density_scale": 0.65,
        "patch_density_bias": 0.0,
        "blob_count": [2, 5],
        "shell_probability": 0.35,
        "noise_amplitude": 0.04,
    },
    "steps": 50,
    "iterations": 160,
    "random_init_count": 4,
    "inner_optim_steps": 125,
    "train_clip_mode": "straight_through_hard",
    "eval_clip_mode": "hard",
    "optimize_params": "m,s",
    "seed": 0,
    "source_selection": "best",
    "source_selection_config": {
        "top_k": 12,
        "temperature": 6.0,
        "score_field": "score_100",
    },
    "source_injection": {
        "enabled": False,
        "every_iterations": 10,
        "start_after_initial_pool": True,
        "max_injections": 0,
    },
    "mutation_std": None,
    "mutation": {
        "probability": 0.0,
        "no_mutation_every_iterations": 0,
        "mutated_inner_optim_steps": None,
        "initial_logit_probability": 1.0,
        "initial_logit_std": 0.0,
        "precheck": {
            "enabled": False,
            "steps": 50,
            "max_attempts": 1,
            "accept_first_life_gate_pass": True,
        },
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
    "evaluation": {
        "continuation_steps": [],
        "stop_on_first_failure": False,
    },
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
    "logging": {
        "inner_log_every": 5,
        "detailed_terms_every": 25,
    },
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
        if isinstance(source.get("initialization"), dict):
            args["source_initialization"] = source["initialization"]

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
        if isinstance(search.get("source_selection_config"), dict):
            args["source_selection_config"] = search["source_selection_config"]
        if isinstance(search.get("source_injection"), dict):
            args["source_injection"] = search["source_injection"]

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
    if isinstance(config.get("evaluation"), dict):
        args["evaluation"] = config["evaluation"]
    if isinstance(config.get("logging"), dict):
        args["logging"] = config["logging"]
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
    result["evaluation"] = _deep_update(DEFAULT_ARGS["evaluation"], result.get("evaluation") or {})
    result["source_initialization"] = _deep_update(DEFAULT_ARGS["source_initialization"], result.get("source_initialization") or {})
    result["source_selection_config"] = _deep_update(DEFAULT_ARGS["source_selection_config"], result.get("source_selection_config") or {})
    result["source_injection"] = _deep_update(DEFAULT_ARGS["source_injection"], result.get("source_injection") or {})
    result["logging"] = _deep_update(DEFAULT_ARGS["logging"], result.get("logging") or {})
    result["rule_randomization"] = _deep_update(DEFAULT_ARGS["rule_randomization"], result.get("rule_randomization") or {})
    result["mutation"] = _deep_update(DEFAULT_ARGS["mutation"], result.get("mutation") or {})
    if result.get("mutation_std") is not None:
        result["mutation"]["initial_logit_std"] = float(result["mutation_std"])
    result["mutation_std"] = float(result["mutation"].get("initial_logit_std", 0.0))
    result["source_initialization"] = _coerce_float_fields(result["source_initialization"])
    result["source_selection_config"] = _coerce_float_fields(result["source_selection_config"])
    result["source_injection"] = _coerce_float_fields(result["source_injection"])
    result["evaluation"] = _coerce_float_fields(result["evaluation"])
    result["logging"] = _coerce_float_fields(result["logging"])
    if isinstance(result["evaluation"].get("continuation_steps"), list):
        result["evaluation"]["continuation_steps"] = [
            int(step) for step in result["evaluation"]["continuation_steps"] if int(step) > 0
        ]
    result["rule_randomization"] = _coerce_float_fields(result["rule_randomization"])
    result["mutation"] = _coerce_float_fields(result["mutation"])
    for key in ("enabled", "start_after_initial_pool"):
        if key in result["source_injection"]:
            result["source_injection"][key] = bool(result["source_injection"].get(key, False))
    if isinstance(result["mutation"].get("precheck"), dict):
        result["mutation"]["precheck"]["enabled"] = bool(result["mutation"]["precheck"].get("enabled", False))
        result["mutation"]["precheck"]["accept_first_life_gate_pass"] = bool(result["mutation"]["precheck"].get("accept_first_life_gate_pass", True))
    result["evaluation"]["stop_on_first_failure"] = bool(result["evaluation"].get("stop_on_first_failure", False))
    for key in ("debug_allow_size_32", "compile_step", "save_initial_states"):
        result[key] = bool(result.get(key, False))
    result["logging"]["inner_log_every"] = max(int(float(result["logging"].get("inner_log_every", 5) or 5)), 1)
    result["logging"]["detailed_terms_every"] = max(int(float(result["logging"].get("detailed_terms_every", 25) or 25)), 1)
    _validate_args(result)
    return argparse.Namespace(**result)

def _validate_args(result: dict[str, Any]) -> None:
    profile = str(result.get("profile", "move_shape_target"))
    if profile not in PROFILE_LOSSES:
        raise ValueError(f"Unknown profile: {profile}")
    score_field = str(result.get("source_selection_config", {}).get("score_field", "score_100"))
    if score_field not in {"score_100", "rank_score_100", "adaptive", "rescue_mixed_score"}:
        raise ValueError(f"Unknown source_selection_config.score_field: {score_field}")
    optimizer_name = str(result.get("optimizer_name", "Adam"))
    if not hasattr(torch.optim, optimizer_name):
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    unsupported = _parse_optimize_params(str(result.get("optimize_params", ""))) - {"m", "s", "T"}
    if unsupported:
        raise ValueError(f"Unsupported differentiable params: {sorted(unsupported)}")
    horizons = result.get("evaluation", {}).get("continuation_steps", [])
    if isinstance(horizons, list) and any(int(value) <= int(result.get("steps", 0)) for value in horizons):
        raise ValueError("continuation_steps must be absolute horizons greater than simulation.steps")
    objective = result.get("objective", {})
    target_mode = str(objective.get("target_mode", "initial_offset")) if isinstance(objective, dict) else "initial_offset"
    if target_mode not in {"initial_offset", "absolute_norm", "absolute_voxel", "none"}:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    model_type = str(result.get("model_type", "legacy_lenia"))
    if model_type != "legacy_lenia":
        raise ValueError(f"Unsupported model type: {model_type}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan 07 Sensorimotor-style VolLenia search MVP.")
    parser.add_argument("--config", type=Path, default=None, help="YAML search config. Explicit CLI flags override config values.")
    parser.add_argument("--profile", choices=("move_shape_target", "rescue_unstable_animal"), default=argparse.SUPPRESS, help="Goal/loss profile used for train loss and hard-eval ranking.")
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
    parser.add_argument("--source-selection", choices=("best", "top_weighted", "top_alive_weighted", "nearest_goal"), default=argparse.SUPPRESS, help="How to pick the archive source for the next candidate.")
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


@dataclass(slots=True)
class SourceSelectionConfig:
    top_k: int = 12
    temperature: float = 6.0
    score_field: str = "score_100"


@dataclass(slots=True)
class SourceInjectionConfig:
    enabled: bool = False
    every_iterations: int = 10
    start_after_initial_pool: bool = True
    max_injections: int = 0


@dataclass(slots=True)
class SourceConfig:
    catalog: Path | None = None
    animal: str = ""
    source_size: int = 64
    initialization: dict[str, Any] = field(default_factory=dict)
    rule_randomization: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MutationConfig:
    probability: float = 0.0
    no_mutation_every_iterations: int = 0
    mutated_inner_optim_steps: int | None = None
    initial_logit_probability: float = 1.0
    initial_logit_std: float = 0.0
    precheck: dict[str, Any] = field(default_factory=dict)
    rule: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationConfig:
    inner_optim_steps: int = 125
    train_clip_mode: str = "straight_through_hard"
    eval_clip_mode: str = "hard"
    optimize_params: str = "m,s"
    compile_step: bool = False
    optimizer_name: str = "Adam"
    optimizer_lr: float = 0.08
    optimizer_initial_lr: float | None = None
    optimizer_param_lr: float | None = None
    max_grad_norm: float | None = None


@dataclass(slots=True)
class EvaluationConfig:
    continuation_steps: list[int] = field(default_factory=list)
    stop_on_first_failure: bool = False


@dataclass(slots=True)
class ExportConfig:
    out: Path = Path("outputs/search_mvp/search")
    export_top_k: int = 3
    save_initial_states: bool = True
    checkpoint_every_iterations: int = 5
    best_every_iterations: int = 1


@dataclass(slots=True)
class LoggingConfig:
    inner_log_every: int = 5
    detailed_terms_every: int = 25


@dataclass(slots=True)
class SearchConfig:
    profile: str = "move_shape_target"
    size: int = 128
    steps: int = 50
    iterations: int = 160
    random_init_count: int = 4
    seed: int = 0
    source_selection: str = "best"
    source_selection_config: SourceSelectionConfig = field(default_factory=SourceSelectionConfig)
    source_injection: SourceInjectionConfig = field(default_factory=SourceInjectionConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    objective: dict[str, Any] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug_allow_size_32: bool = False


def namespace_to_config(args: argparse.Namespace) -> SearchConfig:
    source_selection = dict(args.source_selection_config)
    source_injection = dict(args.source_injection)
    mutation = dict(args.mutation)
    logging = dict(args.logging)
    return SearchConfig(
        profile=args.profile,
        size=args.size,
        steps=args.steps,
        iterations=args.iterations,
        random_init_count=args.random_init_count,
        seed=args.seed,
        source_selection=args.source_selection,
        source_selection_config=SourceSelectionConfig(**source_selection),
        source_injection=SourceInjectionConfig(**source_injection),
        source=SourceConfig(
            catalog=args.catalog,
            animal=args.animal,
            source_size=args.source_size,
            initialization=dict(args.source_initialization),
            rule_randomization=dict(args.rule_randomization),
        ),
        mutation=MutationConfig(**mutation),
        optimization=OptimizationConfig(
            inner_optim_steps=args.inner_optim_steps,
            train_clip_mode=args.train_clip_mode,
            eval_clip_mode=args.eval_clip_mode,
            optimize_params=args.optimize_params,
            compile_step=args.compile_step,
            optimizer_name=args.optimizer_name,
            optimizer_lr=args.optimizer_lr,
            optimizer_initial_lr=args.optimizer_initial_lr,
            optimizer_param_lr=args.optimizer_param_lr,
            max_grad_norm=args.max_grad_norm,
        ),
        evaluation=EvaluationConfig(**dict(args.evaluation)),
        export=ExportConfig(
            out=args.out,
            export_top_k=args.export_top_k,
            save_initial_states=args.save_initial_states,
            checkpoint_every_iterations=args.checkpoint_every_iterations,
            best_every_iterations=args.best_every_iterations,
        ),
        objective=dict(args.objective),
        logging=LoggingConfig(**logging),
        debug_allow_size_32=args.debug_allow_size_32,
    )


def load_search_config(path: Path) -> SearchConfig:
    values = _deep_update(DEFAULT_ARGS, _config_to_args(_load_yaml_config(path)))
    values["config"] = path
    return namespace_to_config(_coerce_args(values))


__all__ = [
    "DEFAULT_ARGS",
    "EvaluationConfig",
    "ExportConfig",
    "LoggingConfig",
    "MutationConfig",
    "OptimizationConfig",
    "SearchConfig",
    "SourceConfig",
    "SourceInjectionConfig",
    "SourceSelectionConfig",
    "_coerce_args",
    "_config_to_args",
    "_deep_update",
    "_load_yaml_config",
    "load_search_config",
    "namespace_to_config",
    "parse_args",
]
