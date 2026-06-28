# Codex prompt — PLAN 07.5 Search refactor / stabilization

You are working in:

```text
D:\projects\VolLenia_Playground
```

The repo already has:

```text
python/scripts/search_sensorimotor_mvp.py
python/vollenia_diff/
configs/search_mvp/
C++ GUI animal catalog bridge
```

## Goal

Refactor and stabilize the current Plan 07 search code before implementing Expanded / Flow Lenia.

Do not implement Expanded/Flow/multi-channel/model changes in this task.

## Read first

```text
plans/PLAN_07_5_search_refactor_stabilization.md
docs/07_sensorimotor_search_design.md
python/scripts/search_sensorimotor_mvp.py
python/vollenia_diff/rollout_losses.py
python/vollenia_diff/metrics.py
configs/search_mvp/move_shape_target/default.yaml
configs/search_mvp/rescue_unstable_animal/default.yaml
```

## High priority implementation tasks

### A. Score field selection

Implement explicit `source_selection_config.score_field`:

```text
score_100
rank_score_100
adaptive
rescue_mixed_score
```

Defaults:

```text
move_shape_target -> score_100
rescue_unstable_animal -> rank_score_100
```

Log selected source score fields and gate state.

### B. Soft active training terms

Add soft active helpers in metrics/loss code.

Replace hard threshold active terms in BPTT training losses with soft versions.

Keep hard active metrics in eval/gates.

### C. Refactor monolithic script

Create `python/vollenia_search/` modules:

```text
config.py
sources.py
archive.py
mutation.py
optimize.py
evaluate.py
export.py
progress.py
profiles.py
```

Keep `python/scripts/search_sensorimotor_mvp.py` as CLI glue.

Do this incrementally. The goal is not perfect architecture; it is smaller testable pieces.

### D. Dataclass config

Introduce dataclass config objects. Keep existing YAML configs working.

Validate unknown profile names, unknown score fields, optimizer names, unsupported differentiable params, invalid target specs.

### E. Target context

Add `TargetSpec` / per-candidate `TargetContext`.

Ensure loss, eval, and export metadata use the same target coordinates.

### F. Dynamic compiled step for trainable params

When optimizing `m/s/T`, provide a compiled dynamic step callable.

If torch.compile fails, produce a clean warning/fallback and write benchmark metadata.

### G. Reduce CPU sync

Add logging config and reduce `.detach().cpu()` calls inside inner BPTT loops.

Only serialize full terms periodically and at final inner step.

## Tests

Add tests for:

```text
score field behavior
soft active gradient
target context consistency
compiled dynamic step correctness
existing config loading
```

## Constraints

- Do not add Hydra.
- Do not implement hot/cold archive storage now.
- Do not change C++ renderer behavior unless required by existing catalog compatibility.
- Keep existing Plan 07 configs runnable.
- Avoid breaking current Python/C++ bridge.

## Final response

After implementation, summarize:

```text
changed modules
new config options
new tests
benchmark results if run
known limitations
commands to run
```
