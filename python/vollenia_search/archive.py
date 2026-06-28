from __future__ import annotations

from typing import Any

import torch

VALID_SCORE_FIELDS = ("score_100", "rank_score_100", "adaptive", "rescue_mixed_score")

def _score_value(item: dict[str, Any]) -> float:
    value = item.get("rank_score_100", item.get("score", float("-inf")))
    return float(value) if isinstance(value, int | float) else float("-inf")

def _selection_score_value(item: dict[str, Any], score_field: str, *, max_horizon: int | None = None) -> float:
    if score_field == "default":
        return _score_value(item)
    if score_field == "rescue_mixed_score":
        rank = float(item.get("rank_score_100", item.get("score", 0.0)) or 0.0)
        raw = float(item.get("score_100", item.get("score", 0.0)) or 0.0)
        longest = max(float(item.get("longest_passed_horizon", 0.0) or 0.0), 0.0)
        horizon = float(max_horizon if max_horizon and max_horizon > 0 else max(int(longest), 1))
        horizon_bonus = 2.0 * float(torch.log1p(torch.tensor(longest)).item()) / float(torch.log1p(torch.tensor(horizon)).item())
        return rank + 0.02 * raw + horizon_bonus
    value = item.get(score_field)
    if isinstance(value, int | float):
        score = float(value)
        return -score if score_field == "loss_total" else score
    return _score_value(item)

def _resolve_selection_score_field(scored: list[dict[str, Any]], requested: str) -> str:
    if requested == "adaptive":
        return "rank_score_100" if any(bool(item.get("life_gate_pass", False)) for item in scored) else "score_100"
    return requested

def _source_entry_id(item: dict[str, Any]) -> str:
    return str(item.get("entry_id") or item.get("id") or item.get("source", {}).get("seed") or "unknown")

def _fresh_sources(archive_memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in archive_memory if item.get("score") == float("-inf") and not item.get("used", False)]

def _scored_sources(archive_memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in archive_memory if item.get("score") != float("-inf")]

def choose_source(
    archive_memory: list[dict[str, Any]],
    mode: str,
    *,
    selection_config: dict[str, Any] | None = None,
    seed: int = 0,
    gate_fail_score_cap: float = 5.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not archive_memory:
        raise ValueError("Archive is empty")
    fresh = _fresh_sources(archive_memory)
    if fresh:
        item = fresh[0]
        return item, {
            "mode": mode,
            "selected_by": "fresh_unscored",
            "source_entry_id": _source_entry_id(item),
            "source_rank_score_100": None,
            "selected_source_score_100": None,
            "selected_source_rank_score_100": None,
            "selected_source_life_gate_pass": None,
            "selected_source_collapse_reason": None,
            "selected_source_longest_passed_horizon": None,
            "selected_source_score_field": None,
        }

    scored = _scored_sources(archive_memory)
    if not scored:
        item = archive_memory[0]
        return item, {"mode": mode, "selected_by": "fallback_first", "source_entry_id": _source_entry_id(item)}

    if mode in {"top_weighted", "top_alive_weighted"}:
        config = selection_config or {}
        top_k = max(int(float(config.get("top_k", 12) or 12)), 1)
        temperature = max(float(config.get("temperature", 6.0) or 6.0), 1.0e-6)
        requested_score_field = str(config.get("score_field", "score_100") or "score_100")
        score_field = _resolve_selection_score_field(scored, requested_score_field)
        pool = scored
        selected_by = "top_weighted"
        if mode == "top_alive_weighted":
            pool = [
                item
                for item in scored
                if bool(item.get("life_gate_pass", False)) and _score_value(item) > float(gate_fail_score_cap)
            ]
            selected_by = "top_alive_weighted"
        if pool:
            max_horizon = max([int(item.get("longest_passed_horizon", 0) or 0) for item in pool] + [1])
            candidates = sorted(pool, key=lambda item: _selection_score_value(item, score_field, max_horizon=max_horizon), reverse=True)[:top_k]
            scores = torch.tensor([_selection_score_value(item, score_field, max_horizon=max_horizon) for item in candidates], dtype=torch.float32)
            weights = torch.softmax((scores - scores.max()) / temperature, dim=0)
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            index = int(torch.multinomial(weights, 1, generator=generator).item())
            item = candidates[index]
            return item, {
                "mode": mode,
                "selected_by": selected_by,
                "source_entry_id": _source_entry_id(item),
                "source_rank_score_100": _score_value(item),
                "selected_source_score_100": item.get("score_100"),
                "selected_source_rank_score_100": item.get("rank_score_100", item.get("score")),
                "selected_source_life_gate_pass": item.get("life_gate_pass"),
                "selected_source_collapse_reason": item.get("collapse_reason"),
                "selected_source_longest_passed_horizon": item.get("longest_passed_horizon"),
                "selected_source_score_field": score_field,
                "top_k": top_k,
                "temperature": temperature,
                "score_field": score_field,
                "requested_score_field": requested_score_field,
                "candidate_ids": [_source_entry_id(candidate) for candidate in candidates],
                "candidate_scores": [float(score) for score in scores.tolist()],
                "selected_index": index,
                "selected_weight": float(weights[index].item()),
            }

    if mode == "nearest_goal":
        item = max(scored, key=_score_value)
        return item, {
            "mode": mode,
            "selected_by": "nearest_goal_fallback_best",
            "source_entry_id": _source_entry_id(item),
            "source_rank_score_100": _score_value(item),
            "selected_source_score_100": item.get("score_100"),
            "selected_source_rank_score_100": item.get("rank_score_100", item.get("score")),
            "selected_source_life_gate_pass": item.get("life_gate_pass"),
            "selected_source_collapse_reason": item.get("collapse_reason"),
            "selected_source_longest_passed_horizon": item.get("longest_passed_horizon"),
            "selected_source_score_field": "rank_score_100",
        }
    item = max(scored, key=_score_value)
    return item, {
        "mode": mode,
        "selected_by": "best",
        "source_entry_id": _source_entry_id(item),
        "source_rank_score_100": _score_value(item),
        "selected_source_score_100": item.get("score_100"),
        "selected_source_rank_score_100": item.get("rank_score_100", item.get("score")),
        "selected_source_life_gate_pass": item.get("life_gate_pass"),
        "selected_source_collapse_reason": item.get("collapse_reason"),
        "selected_source_longest_passed_horizon": item.get("longest_passed_horizon"),
        "selected_source_score_field": "rank_score_100",
    }

def score_100_from_loss(loss_value: float) -> float:
    return 100.0 / (1.0 + max(float(loss_value), 0.0))

def ranked_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(entries, key=lambda item: item.get("rank_score_100", item.get("score", float("-inf"))), reverse=True)
    for rank, entry in enumerate(ranked):
        entry["rank"] = rank
    return ranked

__all__ = [
    "VALID_SCORE_FIELDS",
    "_score_value",
    "_selection_score_value",
    "_source_entry_id",
    "choose_source",
    "ranked_entries",
    "score_100_from_loss",
]
