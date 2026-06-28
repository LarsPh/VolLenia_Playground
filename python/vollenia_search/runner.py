from __future__ import annotations

import copy
import time
from typing import Any

import torch

from vollenia_diff.params import LeniaParams
from vollenia_diff.simulator import require_cuda

from .archive import (
    _score_value,
    _selection_score_value,
    _source_entry_id,
    choose_source,
    ranked_entries,
    score_100_from_loss,
)
from .config import (
    DEFAULT_ARGS,
    _coerce_args,
    _config_to_args,
    _deep_update,
    _load_yaml_config,
    parse_args,
)
from .evaluate import (
    _continuation_horizons,
    evaluate_candidate,
    evaluate_continuation_gates,
    evaluate_life_gate,
    target_context_for_profile,
    target_for_profile,
)
from .export import maybe_write_periodic_catalogs, write_catalog_for_entries, write_outputs
from .mutation import (
    _apply_rule_noise,
    apply_candidate_mutation,
    candidate_mutation_decision,
    disabled_mutation,
    precheck_mutation_candidate,
)
from .optimize import (
    _parse_optimize_params,
    gradient_diagnostics,
    make_optimizer,
    make_raw_params,
    optimize_candidate,
    params_from_raw,
)
from .progress import RatePairColumn, SearchProgress, StatusColumn
from .sources import (
    initial_archive_sources,
    inject_procedural_source,
    procedural_source_state,
    should_inject_source,
    source_from_args,
    source_label,
)
from .targets import target_from_context

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
        entries: list[dict[str, Any]] = []
        export_states: dict[str, torch.Tensor] = {}
        initial_states: dict[str, torch.Tensor] = {}
        export_params: dict[str, LeniaParams] = {}
        search_task = progress.add_task("Search candidates", total=args.iterations, unit="cand", status="starting")
        best_score = float("-inf")
        injections_done = 0

        for iteration in range(int(args.iterations)):
            iter_start = time.perf_counter()
            if should_inject_source(args, iteration=iteration, injections_done=injections_done):
                inject_procedural_source(args, archive_memory, device=device, iteration=iteration, injections_done=injections_done)
                injections_done += 1
            source, selection_metadata = choose_source(
                archive_memory,
                args.source_selection,
                selection_config=args.source_selection_config,
                seed=int(args.seed) + iteration * 7919,
                gate_fail_score_cap=float(args.objective.get("gate_fail_score_cap", 5.0)),
            )
            selected_fresh_source = source.get("score") == float("-inf")
            source["used"] = True
            source_target_context = target_context_for_profile(args.profile, source["state0"], args.objective)
            target = target_from_context(source_target_context)
            current_source = source_label(source)
            mutation_seed = int(args.seed) + iteration * 1000003
            if selected_fresh_source:
                effective_mutation = disabled_mutation(args.mutation)
                inner_steps_used = int(args.inner_optim_steps)
                mutation_decision = {
                    "enabled": True,
                    "applied": False,
                    "reason": "fresh_source_no_mutation",
                    "iteration": int(iteration),
                    "seed": mutation_seed,
                    "inner_steps_used": inner_steps_used,
                    "config": copy.deepcopy(args.mutation),
                    "fresh_injected": bool(source.get("fresh_injected", False)),
                }
            else:
                effective_mutation, mutation_decision, inner_steps_used = candidate_mutation_decision(
                    args.mutation,
                    iteration=iteration,
                    seed=mutation_seed,
                    full_inner_steps=args.inner_optim_steps,
                )
                mutation_seed, precheck_metadata = precheck_mutation_candidate(
                    source["state0"],
                    source["params"],
                    args=args,
                    profile=args.profile,
                    target=target,
                    target_context=source_target_context,
                    mutation=effective_mutation,
                    mutation_decision=mutation_decision,
                    mutation_seed=mutation_seed,
                    progress=progress,
                    search_task=search_task,
                )
                mutation_decision["precheck"] = precheck_metadata
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
                target_context=source_target_context,
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
            eval_horizons = _continuation_horizons(args.evaluation, args.steps)
            eval_task = progress.add_task(
                f"Hard eval {iteration:03d}",
                total=1 + len(eval_horizons),
                unit="eval",
                status=f"primary={args.steps}",
            )
            final_eval, metrics_eval, terms_eval, score = evaluate_candidate(
                learned_state0,
                learned_params,
                profile=args.profile,
                target=target,
                target_context=source_target_context,
                steps=args.steps,
                clip_mode=args.eval_clip_mode,
                compile_step=args.compile_step,
                objective=args.objective,
                evaluation=args.evaluation,
                progress=progress,
                progress_task=eval_task,
            )
            progress.update(eval_task, status=f"score={score:.4g}")
            progress.remove(eval_task)
            candidate_id = f"{args.profile}_{iteration:04d}"
            candidate_source = source.get("source", {})
            source_entry_id = selection_metadata.get("source_entry_id", _source_entry_id(source))
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
                "selection_metadata": selection_metadata,
                "source_entry_id": source_entry_id,
                "source_rank_score_100": selection_metadata.get("source_rank_score_100"),
                "selected_source_score_100": selection_metadata.get("selected_source_score_100"),
                "selected_source_rank_score_100": selection_metadata.get("selected_source_rank_score_100"),
                "selected_source_life_gate_pass": selection_metadata.get("selected_source_life_gate_pass"),
                "selected_source_collapse_reason": selection_metadata.get("selected_source_collapse_reason"),
                "selected_source_longest_passed_horizon": selection_metadata.get("selected_source_longest_passed_horizon"),
                "selected_source_score_field": selection_metadata.get("selected_source_score_field"),
                "source_first_failure_horizon": source.get("first_failure_horizon"),
                "goal_profile": args.profile,
                "params": learned_params.to_catalog_dict(),
                "train_clip_mode": args.train_clip_mode,
                "eval_clip_mode": args.eval_clip_mode,
                "metrics_train": metrics_train,
                "metrics_eval": metrics_eval,
                "descriptor": metrics_eval.get("descriptor", {}),
                "objective_terms": {"train": terms_train, "eval": terms_eval},
                "objective_config": copy.deepcopy(args.objective),
                "target_context": source_target_context.to_json(),
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
                "primary_life_gate_pass": metrics_eval.get("primary_life_gate_pass", metrics_eval.get("life_gate_pass", True)),
                "primary_collapse_reason": metrics_eval.get("primary_collapse_reason", metrics_eval.get("collapse_reason", "ok")),
                "continuation_life_gate_pass": metrics_eval.get("continuation_life_gate_pass", True),
                "continuation_collapse_reason": metrics_eval.get("continuation_collapse_reason", "ok"),
                "first_failure_horizon": metrics_eval.get("first_failure_horizon"),
                "longest_passed_horizon": metrics_eval.get("longest_passed_horizon"),
                "failed_at_final_horizon": metrics_eval.get("failed_at_final_horizon", False),
                "life_topology": metrics_eval.get("life_topology", ""),
                "full_axis_count": metrics_eval.get("full_axis_count", 0),
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
            archive_memory.append({
                "entry_id": candidate_id,
                "state0": learned_state0.detach(),
                "params": learned_params,
                "score": score,
                "rank_score_100": metrics_eval.get("rank_score_100", score),
                "score_100": metrics_eval.get("score_100", score),
                "life_gate_pass": metrics_eval.get("life_gate_pass", True),
                "collapse_reason": metrics_eval.get("collapse_reason", "ok"),
                "first_failure_horizon": metrics_eval.get("first_failure_horizon"),
                "longest_passed_horizon": metrics_eval.get("longest_passed_horizon"),
                "life_topology": metrics_eval.get("life_topology", ""),
                "full_axis_count": metrics_eval.get("full_axis_count", 0),
                "source": candidate_source,
                "source_entry_id": source_entry_id,
                "selection_metadata": selection_metadata,
            })
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
                f"continuation={metrics_eval.get('continuation_life_gate_pass', True)} "
                f"fail_h={metrics_eval.get('first_failure_horizon')} "
                f"topology={metrics_eval.get('life_topology', '')} "
                f"logit_grad_rms={last_grad_diagnostics['logit_grad_rms']:.6g} "
                f"m_grad={last_grad_diagnostics['m_grad']:.6g} "
                f"s_grad={last_grad_diagnostics['s_grad']:.6g} "
                f"T_grad={last_grad_diagnostics['T_grad']:.6g} "
                f"loss={terms_eval.get('loss_total', terms_train.get('loss_total', 0.0)):.6g} {term_status} "
                f"mutation={metrics_train.get('candidate_mutation', {}).get('reason', mutation_status)} "
                f"inner_steps={inner_steps_used} "
                f"max_density={metrics_eval['max_density']:.4f} active={metrics_eval['active_voxels']} "
                f"active_radius={metrics_eval.get('active_body_radius_norm', 0.0):.4f} "
                f"time={entry['timing_seconds']:.3f}s"
            )

        write_outputs(args, entries, export_states, initial_states, export_params, run_started, progress)
        progress.log(f"Wrote search outputs to {args.out}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_ARGS",
    "RatePairColumn",
    "SearchProgress",
    "StatusColumn",
    "_apply_rule_noise",
    "_coerce_args",
    "_config_to_args",
    "_continuation_horizons",
    "_deep_update",
    "_load_yaml_config",
    "_parse_optimize_params",
    "_score_value",
    "_selection_score_value",
    "_source_entry_id",
    "apply_candidate_mutation",
    "candidate_mutation_decision",
    "choose_source",
    "disabled_mutation",
    "evaluate_candidate",
    "evaluate_continuation_gates",
    "evaluate_life_gate",
    "gradient_diagnostics",
    "initial_archive_sources",
    "inject_procedural_source",
    "main",
    "make_optimizer",
    "make_raw_params",
    "optimize_candidate",
    "params_from_raw",
    "parse_args",
    "precheck_mutation_candidate",
    "procedural_source_state",
    "ranked_entries",
    "score_100_from_loss",
    "should_inject_source",
    "source_from_args",
    "source_label",
    "target_context_for_profile",
    "target_for_profile",
    "write_catalog_for_entries",
    "write_outputs",
]


if __name__ == "__main__":
    main()
