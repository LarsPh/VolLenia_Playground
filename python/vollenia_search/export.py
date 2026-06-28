from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from vollenia_diff.export_cpp import tensor_to_f32_file, write_catalog
from vollenia_diff.params import LeniaParams

from .archive import ranked_entries

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
    states = {entry["id"]: initial_states.get(entry["id"], export_states[entry["id"]]) for entry in entries}
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
            "primary_life_gate_pass": entry.get("primary_life_gate_pass", entry.get("life_gate_pass", True)),
            "primary_collapse_reason": entry.get("primary_collapse_reason", entry.get("collapse_reason", "ok")),
            "continuation_steps": entry.get("metrics_eval", {}).get("continuation_steps", []),
            "continuation_life_gate_pass": entry.get("continuation_life_gate_pass", True),
            "continuation_collapse_reason": entry.get("continuation_collapse_reason", "ok"),
            "continuation_results": entry.get("metrics_eval", {}).get("continuation_results", []),
            "first_failure_horizon": entry.get("first_failure_horizon"),
            "longest_passed_horizon": entry.get("longest_passed_horizon"),
            "failed_at_final_horizon": entry.get("failed_at_final_horizon", False),
            "active_body_radius_norm": entry.get("metrics_eval", {}).get(
                "active_body_radius_norm",
                entry.get("descriptor", {}).get("active_body_radius_norm", 0.0),
            ),
            "life_topology": entry.get("life_topology", entry.get("metrics_eval", {}).get("life_topology", "")),
            "full_axis_count": entry.get("full_axis_count", entry.get("metrics_eval", {}).get("full_axis_count", 0)),
            "active_axis_coverage_norm": entry.get("metrics_eval", {}).get(
                "active_axis_coverage_norm",
                entry.get("descriptor", {}).get("active_axis_coverage_norm", []),
            ),
            "active_axis_circular_span_norm": entry.get("metrics_eval", {}).get(
                "active_axis_circular_span_norm",
                entry.get("descriptor", {}).get("active_axis_circular_span_norm", []),
            ),
            "axis_border_mass": entry.get("metrics_eval", {}).get(
                "axis_border_mass",
                entry.get("descriptor", {}).get("axis_border_mass", []),
            ),
            "rank": entry.get("rank", -1),
            "goal_profile": entry["goal_profile"],
            "objective_terms": entry["objective_terms"],
            "objective_config": entry.get("objective_config", {}),
            "train_clip_mode": entry["train_clip_mode"],
            "eval_clip_mode": entry["eval_clip_mode"],
            "export_activation": "raw",
            "catalog_state_role": "learned_initial",
            "eval_final_file": f"snapshots/{entry['id']}_final.f32" if entry["id"] in export_states else "",
            "source": entry["source"],
            "selection_metadata": entry.get("selection_metadata", {}),
            "source_entry_id": entry.get("source_entry_id", ""),
            "source_rank_score_100": entry.get("source_rank_score_100"),
            "selected_source_score_100": entry.get("selected_source_score_100"),
            "selected_source_rank_score_100": entry.get("selected_source_rank_score_100"),
            "selected_source_life_gate_pass": entry.get("selected_source_life_gate_pass"),
            "selected_source_collapse_reason": entry.get("selected_source_collapse_reason"),
            "selected_source_longest_passed_horizon": entry.get("selected_source_longest_passed_horizon"),
            "selected_source_score_field": entry.get("selected_source_score_field"),
            "target_context": entry.get("target_context", {}),
            "source_first_failure_horizon": entry.get("source_first_failure_horizon"),
            "candidate_mutation": entry.get("metrics_train", {}).get("candidate_mutation", {}),
            "mutation_precheck": entry.get("mutation_decision", {}).get("precheck", {}),
            "inner_steps_used": entry.get("inner_steps_used", 0),
            "optimizer": entry.get("optimizer", {}),
            "gradient_diagnostics": entry.get("gradient_diagnostics", {}),
            "learned_initial_file": f"initials/{entry['id']}_initial.f32" if args.save_initial_states and entry["id"] in initial_states else "",
        }
        for entry in entries
    }
    initial_subset: dict[str, torch.Tensor] = {}
    snapshots_dir = out_dir / "snapshots"
    for entry in entries:
        state = export_states.get(entry["id"])
        if state is not None:
            tensor_to_f32_file(state, snapshots_dir / f"{entry['id']}_final.f32")
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
                "primary_life_gate_pass": entry.get("primary_life_gate_pass", entry.get("life_gate_pass", True)),
                "primary_collapse_reason": entry.get("primary_collapse_reason", entry.get("collapse_reason", "ok")),
                "continuation_steps": entry.get("metrics_eval", {}).get("continuation_steps", []),
                "continuation_life_gate_pass": entry.get("continuation_life_gate_pass", True),
                "continuation_collapse_reason": entry.get("continuation_collapse_reason", "ok"),
                "first_failure_horizon": entry.get("first_failure_horizon"),
                "longest_passed_horizon": entry.get("longest_passed_horizon"),
                "failed_at_final_horizon": entry.get("failed_at_final_horizon", False),
                "active_body_radius_norm": entry.get("metrics_eval", {}).get(
                    "active_body_radius_norm",
                    entry.get("descriptor", {}).get("active_body_radius_norm", 0.0),
                ),
                "life_topology": entry.get("life_topology", entry.get("metrics_eval", {}).get("life_topology", "")),
                "full_axis_count": entry.get("full_axis_count", entry.get("metrics_eval", {}).get("full_axis_count", 0)),
                "active_axis_coverage_norm": entry.get("metrics_eval", {}).get(
                    "active_axis_coverage_norm",
                    entry.get("descriptor", {}).get("active_axis_coverage_norm", []),
                ),
                "active_axis_circular_span_norm": entry.get("metrics_eval", {}).get(
                    "active_axis_circular_span_norm",
                    entry.get("descriptor", {}).get("active_axis_circular_span_norm", []),
                ),
                "axis_border_mass": entry.get("metrics_eval", {}).get(
                    "axis_border_mass",
                    entry.get("descriptor", {}).get("axis_border_mass", []),
                ),
                "rank": entry.get("rank", -1),
                "goal_profile": entry["goal_profile"],
                "state_role": "learned_initial",
                "catalog_state_role": "learned_initial",
                "eval_final_file": f"../snapshots/{entry['id']}_final.f32",
                "final_cells_file": f"../cells/{entry['id']}.f32",
                "source": entry["source"],
                "selection_metadata": entry.get("selection_metadata", {}),
                "source_entry_id": entry.get("source_entry_id", ""),
                "source_rank_score_100": entry.get("source_rank_score_100"),
                "selected_source_score_100": entry.get("selected_source_score_100"),
                "selected_source_rank_score_100": entry.get("selected_source_rank_score_100"),
                "selected_source_life_gate_pass": entry.get("selected_source_life_gate_pass"),
                "selected_source_collapse_reason": entry.get("selected_source_collapse_reason"),
                "selected_source_longest_passed_horizon": entry.get("selected_source_longest_passed_horizon"),
                "selected_source_score_field": entry.get("selected_source_score_field"),
                "target_context": entry.get("target_context", {}),
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
                "primary_life_gate_pass",
                "primary_collapse_reason",
                "continuation_life_gate_pass",
                "continuation_collapse_reason",
                "first_failure_horizon",
                "longest_passed_horizon",
                "failed_at_final_horizon",
                "active_body_radius_norm",
                "life_topology",
                "full_axis_count",
                "active_axis_coverage_norm",
                "active_axis_circular_span_norm",
                "axis_border_mass",
                "goal_profile",
                "source_slug",
                "source_entry_id",
                "selected_source_score_field",
                "selected_source_score_100",
                "selected_source_rank_score_100",
                "selected_source_life_gate_pass",
                "selected_source_collapse_reason",
                "selected_source_longest_passed_horizon",
                "selection_mode",
                "selection_selected_by",
                "mutation_precheck_pass",
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
                "primary_life_gate_pass": entry.get("primary_life_gate_pass", entry.get("life_gate_pass", True)),
                "primary_collapse_reason": entry.get("primary_collapse_reason", entry.get("collapse_reason", "ok")),
                "continuation_life_gate_pass": entry.get("continuation_life_gate_pass", True),
                "continuation_collapse_reason": entry.get("continuation_collapse_reason", "ok"),
                "first_failure_horizon": entry.get("first_failure_horizon"),
                "longest_passed_horizon": entry.get("longest_passed_horizon"),
                "failed_at_final_horizon": entry.get("failed_at_final_horizon", False),
                "active_body_radius_norm": entry.get("metrics_eval", {}).get(
                    "active_body_radius_norm",
                    entry.get("descriptor", {}).get("active_body_radius_norm", 0.0),
                ),
                "life_topology": entry.get("life_topology", entry.get("metrics_eval", {}).get("life_topology", "")),
                "full_axis_count": entry.get("full_axis_count", entry.get("metrics_eval", {}).get("full_axis_count", 0)),
                "active_axis_coverage_norm": json.dumps(entry.get("metrics_eval", {}).get(
                    "active_axis_coverage_norm",
                    entry.get("descriptor", {}).get("active_axis_coverage_norm", []),
                )),
                "active_axis_circular_span_norm": json.dumps(entry.get("metrics_eval", {}).get(
                    "active_axis_circular_span_norm",
                    entry.get("descriptor", {}).get("active_axis_circular_span_norm", []),
                )),
                "axis_border_mass": json.dumps(entry.get("metrics_eval", {}).get(
                    "axis_border_mass",
                    entry.get("descriptor", {}).get("axis_border_mass", []),
                )),
                "goal_profile": entry["goal_profile"],
                "source_slug": entry["source"].get("slug", entry["source"].get("kind", "")),
                "source_entry_id": entry.get("source_entry_id", ""),
                "selected_source_score_field": entry.get("selected_source_score_field", ""),
                "selected_source_score_100": entry.get("selected_source_score_100"),
                "selected_source_rank_score_100": entry.get("selected_source_rank_score_100"),
                "selected_source_life_gate_pass": entry.get("selected_source_life_gate_pass"),
                "selected_source_collapse_reason": entry.get("selected_source_collapse_reason"),
                "selected_source_longest_passed_horizon": entry.get("selected_source_longest_passed_horizon"),
                "selection_mode": entry.get("selection_metadata", {}).get("mode", ""),
                "selection_selected_by": entry.get("selection_metadata", {}).get("selected_by", ""),
                "mutation_precheck_pass": entry.get("mutation_decision", {}).get("precheck", {}).get("pass", ""),
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
        f"- evaluation: `{json.dumps(args.evaluation, sort_keys=True)}`",
        f"- source_selection: `{args.source_selection}`",
        f"- source_selection_config: `{json.dumps(args.source_selection_config, sort_keys=True)}`",
        f"- source_injection: `{json.dumps(args.source_injection, sort_keys=True)}`",
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
        f"- best_life_topology: `{top[0].get('life_topology', 'n/a') if top else 'n/a'}`",
        f"- best_full_axis_count: `{top[0].get('full_axis_count', 'n/a') if top else 'n/a'}`",
        f"- best_continuation_life_gate_pass: `{top[0].get('continuation_life_gate_pass', 'n/a') if top else 'n/a'}`",
        f"- best_continuation_collapse_reason: `{top[0].get('continuation_collapse_reason', 'n/a') if top else 'n/a'}`",
        f"- best_first_failure_horizon: `{top[0].get('first_failure_horizon', 'n/a') if top else 'n/a'}`",
        f"- best_longest_passed_horizon: `{top[0].get('longest_passed_horizon', 'n/a') if top else 'n/a'}`",
        "",
        "## Archive storage recommendation",
        "",
        "- Keep a small GPU hot set for source candidates likely to be sampled in the next few iterations.",
        "- Move older dense tensors to CPU memory while preserving scalar archive metadata in JSON for ranking and filtering.",
        "- Write disk snapshots for exported, checkpointed, or long-horizon candidates; avoid disk writes for every failed near-miss until Expanded/Flow memory pressure is measured.",
        "- Defer a formal TensorRef/GpuTensorRef/CpuTensorRef/DiskTensorRef implementation until the next model layer defines expected state size and channel count.",
        "",
        "Catalog can be opened from the C++ Animal Catalog file picker.",
    ]
    if args.debug_allow_size_32 and args.size == 32:
        summary.append("\nDebug-only run: size 32 is not a Plan 07 search acceptance size.")
    (args.out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    progress.update(export_task, advance=1, status="done")
    progress.remove(export_task)

__all__ = ["maybe_write_periodic_catalogs", "write_catalog_for_entries", "write_outputs"]
