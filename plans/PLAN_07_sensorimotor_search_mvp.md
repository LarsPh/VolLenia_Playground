# PLAN 07 — Sensorimotor-style differentiable search MVP, no obstacles

## Purpose

Build the first real VolLenia search loop. This milestone should not be just cleanup and should not jump all the way to MAP-Elites or resource environments. It should create a minimal Sensorimotor-style loop:

```text
catalog / procedural source candidates
  -> evaluate behavior descriptors
  -> choose/search goals
  -> inner BPTT / gradient refinement
  -> hard-clamp evaluation
  -> export candidates to the C++ renderer bridge
```

This plan has two equally important goals:

1. Test a Sensorimotor Lenia-style goal optimization loop with target movement and shape maintenance.
2. Expand the stable Lenia3D animal neighborhood: find variants near stable animals and try to rescue nearby stable variants from known unstable Lenia3D animals.

Do **not** add obstacle channels in this milestone. Obstacle/resource fields are later curriculum stages.

## Reference repos

Current project:

```text
D:\projects\VolLenia_Playground
```

Sensorimotor reference:

```text
D:\projects\sensorimotor-lenia-search
```

Lenia3D reference:

```text
D:\projects\Lenia3D
```

Use Sensorimotor Lenia to understand the algorithmic pattern:

```text
random initialization archive
sample target goal
select archived source closest to target
copy or mutate source
inner Adam/BPTT optimization
multi-rollout evaluation
archive reached goal
```

Do not copy its repo structure wholesale. VolLenia has different constraints: 3D volumes, C++ renderer bridge, Lenia3D catalog, and eventual environment curriculum.

## Non-goals

- No obstacle channel.
- No resource/nutrient channel.
- No multi-channel Lenia.
- No full MAP-Elites yet.
- No Flow-Lenia yet.
- No C++ headless worker yet.
- No custom CUDA autograd.
- No gradient optimization of discrete `kn`, `gn`, resolution, or volume dimensionality.
- Do not optimize `R` by gradient with the current hard shell-indexed kernel. Treat `R` as random/mutation/search parameter unless a smooth radial-basis kernel is added later.

## Required semantic fixes included in this milestone

These are not unrelated cleanup. They are prerequisites for a meaningful search loop.

### 1. Catalog simulation dimensions

Extend catalog schema:

```json
{
  "dims": [cells_nx, cells_ny, cells_nz],
  "simulation_dims": [sim_nx, sim_ny, sim_nz],
  "resolution_policy": "native|cropped|scaled"
}
```

Semantics:

- `dims`: actual `.f32` tensor dimensions.
- `simulation_dims`: original simulation/world/canvas dimensions.
- `resolution_policy="native"`: full simulation state; usually `dims == simulation_dims`.
- `resolution_policy="cropped"`: cropped cells to be center-padded into `simulation_dims`.
- `resolution_policy="scaled"`: already resampled/scaled cells.

For Lenia3D reference animals, set:

```json
"simulation_dims": [64, 64, 64],
"resolution_policy": "cropped"
```

For PyTorch exported full states, set:

```json
"simulation_dims": dims,
"resolution_policy": "native"
```

C++ GUI behavior:

- `Load native state + native rule`: if `simulation_dims` is cubic, set `lenia_config_.resolution` to that native size before loading cells.
- `Load scaled state + scaled rule`: keep the currently selected simulation resolution, resample cells with current scale, and optionally scale `R`.
- Show both `cells dims` and `simulation dims` in the Animal Catalog UI.
- Preserve backward compatibility for catalogs missing `simulation_dims`.

### 2. Clamp/update modes tied to search

Add or complete these modes in the PyTorch backend:

```text
hard:
  forward = clamp(A + delta, 0, 1)
  closest to C++ / original Lenia

straight_through_hard:
  forward = hard clamp
  backward = identity-like surrogate
  biased gradient estimator, but useful for optimization

none:
  debug only; can leave physical [0,1]

soft_tanh:
  experimental; smooth bounded surrogate, changes dynamics
```

Search default:

```text
train modes to compare:
  hard
  straight_through_hard

evaluation mode:
  hard only
```

Every exported candidate must record:

```text
train_clip_mode
eval_clip_mode
export_activation
```

### 3. Metrics and physical density

For `hard` and `straight_through_hard`, loss/metrics should operate directly on the forward physical density `A_t` in `[0,1]`.

Add `density_view(state, mode="raw|clamp|sigmoid|softplus")` only for debug/surrogate states, not to silently decouple the main search objective from the actual forward dynamics.

For nonphysical/debug modes, record both raw-state and physical-density summaries.

## Metrics and normalized descriptors

Existing metrics should be reused rather than wasted:

```text
mass
mean_density
max_density
active_voxels
center_of_mass
second_moment
covariance_eigvals
anisotropy
border_mass
target_distance
```

Add normalized descriptors for search:

```text
volume_voxels = nx * ny * nz
min_dim = min(nx, ny, nz)
mass_fraction = mass / volume_voxels
active_fraction = active_voxels / volume_voxels
com_norm = center_of_mass / (shape - 1)
second_moment_norm = second_moment / (min_dim * min_dim)
body_radius = sqrt(second_moment + eps)
target_distance_body = target_distance / (body_radius + eps)
target_distance_norm = target_distance / min_dim
mass_ratio = mass / (initial_mass + eps)
active_ratio = active_voxels / (initial_active_voxels + eps)
compactness_ratio = second_moment / (initial_second_moment + eps)
anisotropy = max_cov_eig / min_cov_eig
border_mass = border_mass fraction
```

The resolution-invariant profiles should prefer ratios and normalized units. Absolute metrics may still be logged for debugging.

## Goal profiles for this milestone

Implement a small goal-profile registry. Do not hard-code everything in one script.

### Profile A — `move_shape_target`

This is the Sensorimotor-style target movement task without obstacles.

Purpose:

```text
Can a candidate be optimized to move toward a target while remaining visible, compact, and roughly blob-like?
```

Default source:

```text
procedural seed or a stable imported/scaled Lenia3D animal
```

Target:

```text
target center = initial COM + normalized displacement
or sampled target COM in normalized volume coordinates
```

Loss terms:

```text
L_target_mask:
  final density should overlap a soft target sphere mask

L_com:
  final COM near target center

L_mass_ratio:
  mass stays in [mass_min_ratio, mass_max_ratio] relative to initial mass

L_compact_ratio:
  second_moment stays in a reasonable interval relative to initial

L_anisotropy:
  penalize cylinder/sheet-like shapes above a threshold

L_border:
  avoid periodic-boundary/edge cheating

L_visibility:
  active_voxels and max_density remain visible
```

Notes:

- Sensorimotor Lenia uses target disk loss. VolLenia should support target sphere loss.
- To avoid pure target-shape painting, keep mass/compactness/anisotropy terms active.
- Evaluate final candidate in hard mode even if trained with straight-through.

### Profile B — `maintain_animal_profile`

Purpose:

```text
Search around a known stable Lenia3D animal for nearby stable variants.
```

Default source:

```text
stable Lenia3D animals from current catalog
```

Optimization target:

```text
stay alive and visually coherent over a window
keep normalized mass, active fraction, compactness, anisotropy, and border mass in reasonable ranges
```

Loss terms:

```text
mass_ratio_window_loss
active_ratio_window_loss
compactness_ratio_window_loss
anisotropy_ceiling_loss
border_loss
visibility_loss
optional recurrence / final-vs-mid metric later
```

This profile is not trying to move. It is the baseline “what is a VolLenia organism?” profile.

### Profile C — `rescue_unstable_animal`

Purpose:

```text
Given a known unstable Lenia3D animal, search for a nearby stable variant.
```

Known unstable/rescale-fragile starting list from manual exploration:

```text
oveme torus
hexahedrome inversus
Foraminome lithos [4As1l]
granome vetilans
stylomembranome limus
sytlomembranome lithos [2MeS2l]
stylomembranome_tardus
stylomembranome_tardus_2
planomembranome_saliens
planomembranome_lithos
planomembranome_inversus
planomembranome_vagus
animal_32
animal_33
Animal #34
Animal #35
```

Search strategy:

```text
load native state + native rule
create candidate variants by small random/mutation changes to continuous params
optionally optimize initial logits around imported cells
optionally optimize m/s/T/b, not R by gradient in the first pass
evaluate survival/visibility/compactness over hard-clamp rollout
```

This profile can reuse Profile B’s stability loss but should log the starting animal and mutation distance.

MVP requirement:

If implementing all three profiles is too much, implement Profile A and Profile B. Profile C can be a thin wrapper around B with an unstable animal allow-list.

## Search loop MVP

Add a Python script, suggested:

```text
python/scripts/search_sensorimotor_mvp.py
```

Suggested arguments:

```text
--profile move_shape_target|maintain_animal_profile|rescue_unstable_animal
--catalog configs/lenia3d_reference/animals.json
--animal <slug-or-index-or-display-substring>
--size 32|64
--steps 32
--iterations 10
--random-init-count 5
--inner-optim-steps 20
--train-clip-mode hard|straight_through_hard
--eval-clip-mode hard
--optimize-init-logits
--optimize-params m,s,T,b
--out outputs/search_mvp/<run_name>
--export-every-best
--seed 0
```

The first version may run one source animal/profile rather than a huge catalog batch.

### Archive structure

Implement a simple archive, not full MAP-Elites yet.

Each entry should store:

```json
{
  "id": "...",
  "source": {...},
  "goal_profile": "...",
  "params": {...},
  "train_clip_mode": "hard|straight_through_hard",
  "eval_clip_mode": "hard",
  "metrics_train": {...},
  "metrics_eval": {...},
  "descriptor": {...},
  "score": 0.0,
  "rank": 0,
  "artifact_paths": {...}
}
```

Archive operations for MVP:

```text
1. seed archive from source candidates
2. mutate or copy a source candidate
3. optimize candidate by BPTT
4. evaluate candidate under hard mode
5. insert candidate if visible/alive and score improves
6. export top K to C++ catalog
```

### Source selection

Implement simple modes:

```text
random:
  choose archive candidate uniformly

best:
  choose highest score candidate

nearest_goal:
  choose candidate with descriptor closest to sampled goal
```

For this milestone, `best` or `nearest_goal` is enough.

## Optimization objects

First pass:

```text
optimize initial logits
```

Second pass, if feasible in same milestone:

```text
optimize m, s, T, b
```

Do not gradient-optimize:

```text
R
kn
gn
resolution
```

`m/s/T/b` parameterization:

```text
m: sigmoid/logit or clamp after optimizer step into valid range
s: softplus + epsilon
T: softplus + min_T
b: softplus per shell or sigmoid/clamp, then export raw positive weights
```

After any parameter optimization, export optimized params to catalog and metrics.

## Rollout loss

Add or extend:

```text
python/vollenia_diff/rollout_losses.py
```

Core functions:

```python
rollout_collect(sim, state0, steps, sample_interval=1)
rollout_loss_full_bptt(sim, state0, profile, steps, weights)
rollout_loss_chunked(sim, state0, profile, steps, chunk_size, detach_between_chunks=True)
```

MVP can use full BPTT for small 32^3 runs. Chunked mode can be implemented as a simple helper but does not have to be the default.

Do not call `backward(retain_graph=True)` every step unless explicitly testing. Accumulate scalar loss and call `backward()` once per rollout or chunk.

## Output and C++ bridge

Write outputs under:

```text
outputs/search_mvp/<run_name>/
  archive.json
  metrics.json
  candidates.csv
  summary.md
  catalog.json
  cells/*.f32
  snapshots/*.f32
```

The exported `catalog.json` must include:

```text
simulation_dims
resolution_policy
score
rank
goal_profile
objective_terms
train_clip_mode
eval_clip_mode
source animal / source seed metadata
```

The C++ app should be able to open this catalog via the existing Animal Catalog file picker.

## Tests

Add tests for:

```text
catalog simulation_dims writer/parser compatibility
hard and straight-through output range
nonzero gradients under hard or straight-through on a short rollout if possible
normalized descriptor finite values across 2D/3D
Profile A loss finite
Profile B loss finite
exported search catalog loads as manifest JSON and references real .f32 files
```

If hard-clamp gradient is zero for some tests, use straight-through for the gradient-nonzero assertion and hard for output-range/eval semantics.

## Acceptance criteria

Minimum pass:

```text
uv run python -m pytest python/tests

uv run python python/scripts/search_sensorimotor_mvp.py \
  --profile move_shape_target \
  --size 32 \
  --steps 16 \
  --iterations 3 \
  --inner-optim-steps 3 \
  --train-clip-mode hard \
  --out outputs/search_mvp/smoke_move_hard

uv run python python/scripts/search_sensorimotor_mvp.py \
  --profile maintain_animal_profile \
  --catalog configs/lenia3d_reference/animals.json \
  --animal "Diguttome" \
  --size 64 \
  --steps 16 \
  --iterations 3 \
  --inner-optim-steps 3 \
  --train-clip-mode straight_through_hard \
  --out outputs/search_mvp/smoke_maintain_animal
```

Expected outputs:

```text
archive.json exists
metrics.json exists
summary.md exists
catalog.json exists
cells/*.f32 exists
C++ GUI can open catalog.json and render exported candidates
```

## Implementation notes

- Keep scripts small and inspectable. Avoid a huge framework on the first pass.
- Preserve Plan 06 bridge behavior.
- Do not break Lenia3D reference animal loading.
- Prefer simple JSON configs / CLI args over hidden global constants.
- Log enough metadata that a candidate can be replayed later.
- If a design choice affects search semantics, document it in `summary.md` and code comments.
