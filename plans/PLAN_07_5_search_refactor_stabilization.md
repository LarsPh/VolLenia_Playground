# PLAN 07.5 — Search refactor and stabilization before Expanded / Flow

## Goal

Refactor and stabilize the current Plan 07 search code so that it can survive the upcoming model expansion:

```text
legacy Lenia -> Expanded Lenia -> Flow Lenia
```

This plan does not introduce Flow, multi-channel state, multi-kernel state, or resource fields. It improves the search framework, metrics/loss semantics, score selection, target semantics, config typing, and profiling hooks.

The latest search script is useful but too monolithic. The next model step should not be built on a single 1000+ line script.

## Non-goals

Do not implement:

```text
- Expanded Lenia
- Flow Lenia
- multi-channel state
- resource fields
- MAP-Elites
- hot/cold archive storage
- neural update rules
- C++ headless worker
```

Do not change the working C++ GUI/renderer behavior except where catalog metadata compatibility requires it.

## Required changes

### 1. Clarify source-selection score fields

Current search has two score layers:

```text
score_100      = raw short-horizon objective score
rank_score_100 = gate-aware ranking score
```

Add explicit config:

```yaml
source_selection_config:
  score_field: score_100 | rank_score_100 | adaptive | rescue_mixed_score
```

Suggested defaults:

```yaml
move_shape_target:
  score_field: score_100

rescue_unstable_animal:
  score_field: rank_score_100
```

Adaptive mode:

```text
if no archive entry has life_gate_pass:
    use score_100
else:
    use rank_score_100
```

Optional rescue mixed score:

```text
rescue_mixed_score =
    rank_score_100
  + 0.02 * score_100
  + 2.0 * log1p(longest_passed_horizon) / log1p(max_horizon)
```

Add selected-source diagnostics:

```text
selected_source_score_100
selected_source_rank_score_100
selected_source_life_gate_pass
selected_source_collapse_reason
selected_source_longest_passed_horizon
selected_source_score_field
```

### 2. Use soft active terms in training loss, hard terms in eval

Keep hard active metrics for evaluation:

```python
active_voxels = (state > threshold).sum()
```

But training loss should not depend on non-differentiable threshold counts.

Add helpers:

```python
soft_active_field(state, threshold=0.05, sharpness=40.0)
soft_active_fraction(state, threshold=0.05, sharpness=40.0)
soft_active_count(state, threshold=0.05, sharpness=40.0)
soft_active_ratio(final, initial, threshold=0.05, sharpness=40.0)
```

In train losses:

```text
active_ratio        -> soft_active_ratio
active_fraction     -> soft_active_fraction
absolute_occupancy  -> uses mass_fraction + soft_active_fraction
visibility          -> uses soft active + max density
```

In hard eval / diagnostics, keep:

```text
active_voxels
active_fraction
topology classifier
active axis spans
```

### 3. Do not add differentiable topology yet

Keep topology classifier as eval-only.

Do not add topology losses into BPTT in this plan.

### 4. Split `search_sensorimotor_mvp.py`

Create:

```text
python/vollenia_search/
  __init__.py
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

Keep:

```text
python/scripts/search_sensorimotor_mvp.py
```

as a thin CLI wrapper.

Responsibilities:

```text
config.py     dataclass config, YAML load, CLI override merge, validation
sources.py    procedural source, catalog source, source injection
archive.py    archive entries, source selection, score field handling
mutation.py   rule noise, initial noise, mutation decision, precheck
optimize.py   BPTT inner loop, optimizer creation, gradient diagnostics
evaluate.py   hard eval, continuation eval, life gates, topology gates
export.py     catalog export, CSV, summaries
progress.py   Rich progress UI wrappers
profiles.py   profile registry and profile-specific defaults
```

Do not aim for perfect abstraction. Aim for smaller files with testable functions.

### 5. Introduce dataclass config schema

Replace weak `DEFAULT_ARGS + dict deep_update + coercion` with dataclasses.

Suggested structure:

```python
@dataclass
class SourceConfig: ...
@dataclass
class RuleRandomizationConfig: ...
@dataclass
class MutationConfig: ...
@dataclass
class SourceSelectionConfig: ...
@dataclass
class OptimizationConfig: ...
@dataclass
class EvaluationConfig: ...
@dataclass
class ExportConfig: ...
@dataclass
class ObjectiveConfig: ...
@dataclass
class SearchConfig: ...
```

Validation should catch:

```text
unknown profile names
unknown score fields
invalid optimizer names
unsupported differentiable params
invalid continuation horizons
invalid target specs
invalid model type
```

Do not introduce Hydra yet. Keep YAML support compatible with existing configs.

### 6. Fix target definition semantics

Introduce target spec:

```python
@dataclass
class TargetStage:
    at_step: int | None
    at_step_fraction: float | None
    offset_norm_zyx: list[float]
    weight_scale: dict[str, float]

@dataclass
class TargetSpec:
    mode: Literal["initial_offset", "absolute_norm", "absolute_voxel", "none"]
    offset_norm_zyx: list[float]
    stages: list[TargetStage]
```

Per candidate:

```python
target_context = make_target_context(profile, state0, objective)
loss(..., target_context=target_context)
evaluate(..., target_context=target_context)
export metadata includes target_context
```

Rules:

```text
- targets depending on initial state must be generated per candidate
- staged target loss and eval descriptor must use the same target coordinates
- exported candidate metadata must include final target voxel and normalized target
```

### 7. Compile dynamic step for trainable m/s/T

Current dynamic m/s/T path bypasses compiled simulator step.

Add a dynamic compiled step callable:

```python
def dynamic_step_fn(state, kernel_hat, m, s, T):
    return lenia_step_tensor(state, kernel_hat, shape, m, s, T, gn, clip_id)
```

Cache by:

```text
shape
dtype
device
gn
clip mode
compile backend/mode
```

Benchmark:

```text
eager train step with tensor m/s/T
compiled dynamic step
compiled rollout chunk if implemented
```

Record max absolute difference.

### 8. Reduce CPU sync in inner BPTT

Add config:

```yaml
logging:
  inner_log_every: 5
  detailed_terms_every: 25
```

During most inner steps:

```text
- keep loss/terms on GPU
- do not convert every term to Python floats
- only sync selected summaries every inner_log_every
- full term JSON only final step or detailed_terms_every
```

Benchmark:

```text
inner BPTT steps/sec
wall time per candidate
```

### 9. Archive memory policy: design only

Do not fully implement hot/cold archive in this plan unless it becomes trivial after refactor.

Add design notes and interfaces:

```python
class TensorRef: ...
class GpuTensorRef(TensorRef): ...
class CpuTensorRef(TensorRef): ...
class DiskTensorRef(TensorRef): ...
```

The actual hot/cold policy can be a later plan, because Expanded/Flow will change memory pressure.

## Tests

Add or update tests for:

```text
- score field selection: score_100 vs rank_score_100 vs adaptive
- soft active terms have nonzero gradients near threshold
- hard active descriptors still appear in eval summaries
- target context is per-candidate when using initial_offset
- staged target loss and eval target match
- dynamic compiled step matches eager step within tolerance
- CLI wrapper can load existing move/rescue YAML configs
```

## Acceptance criteria

```text
1. python -m pytest python/tests passes.
2. Existing search configs still run.
3. search_sensorimotor_mvp.py is mostly CLI glue.
4. Source selection score field is explicit and logged.
5. Training loss no longer depends on hard active threshold terms.
6. Target context is consistent between loss, evaluation, and export.
7. Dynamic m/s/T compiled step works or cleanly falls back with benchmark note.
8. Inner BPTT logging causes less CPU synchronization.
```

## Suggested commit

```powershell
git add python docs configs
git commit -m "search: refactor sensorimotor MVP and stabilize training semantics"
```
