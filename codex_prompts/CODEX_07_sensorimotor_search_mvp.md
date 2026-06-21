# Codex prompt — PLAN 07 Sensorimotor-style differentiable search MVP

You are working in:

```text
D:\projects\VolLenia_Playground
```

Reference repo available locally:

```text
D:\projects\sensorimotor-lenia-search
```

Use Sensorimotor Lenia as an algorithm reference, not as code to copy wholesale.

## Goal

Implement the first VolLenia differentiable search MVP. Scope:

- no obstacle channel;
- no resource channel;
- no multi-channel VolLenia;
- include target movement + shape maintenance search;
- include Lenia3D animal neighborhood stability search;
- export search results to the existing C++ renderer bridge.

This milestone should combine required semantic fixes with the first search loop. Do not implement isolated cleanup without connecting it to the search workflow.

## Must implement

### A. Catalog resolution semantics

Extend Python exporter/catalog schema and C++ parser/UI behavior:

- `dims`: actual `.f32` cell tensor dimensions.
- `simulation_dims`: original simulation volume dimensions.
- `resolution_policy`: `native|cropped|scaled`.

Rules:

- PyTorch full-state exports: `dims == simulation_dims`, `resolution_policy="native"`.
- Lenia3D reference animals: set `simulation_dims=[64,64,64]`, `resolution_policy="cropped"`.
- C++ native load should use native simulation resolution when cubic.
- Scaled load should keep current selected resolution.
- Preserve backward compatibility when old catalogs omit these fields.

Update UI labels:

- `Load native state + native rule`
- `Load scaled state + scaled rule`
- Show `cells dims` and `simulation dims`.

### B. Clamp/update modes

Extend PyTorch `LeniaSimulator` clip/update modes:

- `hard`: forward `torch.clamp(x, 0, 1)`.
- `none`: existing debug mode.
- `straight_through_hard`: hard forward clamp with identity-like backward surrogate.
- Optional `soft_tanh`: smooth bounded surrogate.

Search default:

- train mode can be `hard` or `straight_through_hard`.
- eval mode must be `hard`.

Record train/eval mode in all metrics and exported catalog metadata.

### C. Metrics and normalized descriptors

Reuse existing metrics and add normalized descriptors:

- mass_fraction
- active_fraction
- com_norm
- second_moment_norm
- body_radius
- target_distance_body
- target_distance_norm
- mass_ratio vs initial
- active_ratio vs initial
- compactness_ratio vs initial
- anisotropy
- border_mass

Metrics/loss for hard and straight-through should operate on physical forward density, not an unrelated export activation.

Add a `density_view(state, mode="raw|clamp|sigmoid|softplus")` helper for debug/surrogate modes, but do not silently use it in the main hard/ST search path.

### D. Goal profiles

Implement a goal profile registry or clear functions.

#### `move_shape_target`

Sensorimotor-style target movement without obstacles.

Loss terms:

- soft target sphere mask loss or COM target loss;
- mass ratio window;
- compactness ratio window;
- anisotropy ceiling;
- border loss;
- visibility/active voxel loss.

#### `maintain_animal_profile`

Search around a known stable Lenia3D animal.

Loss terms:

- mass ratio window across rollout;
- active ratio window;
- compactness ratio window;
- anisotropy ceiling;
- border loss;
- visibility loss.

#### `rescue_unstable_animal` optional

Thin wrapper around `maintain_animal_profile` with an unstable animal allow-list. If time is short, implement Profile A and B first.

### E. Search script

Add:

```text
python/scripts/search_sensorimotor_mvp.py
```

It should support CLI args similar to:

```text
--profile move_shape_target|maintain_animal_profile|rescue_unstable_animal
--catalog configs/lenia3d_reference/animals.json
--animal <slug-or-index-or-substring>
--size 32|64
--steps 16|32|64
--iterations N
--inner-optim-steps N
--train-clip-mode hard|straight_through_hard
--eval-clip-mode hard
--optimize-init-logits
--optimize-params m,s,T,b
--out outputs/search_mvp/<run_name>
--seed 0
```

For MVP, optimizing initial logits is required. Optimizing m/s/T/b is optional but desirable if it is not too large a change.

Do not gradient-optimize R, kn, gn, resolution.

### F. Archive and export

Implement a small JSON archive, not full MAP-Elites.

Archive entries should store:

- source metadata;
- goal profile;
- params;
- train/eval clip mode;
- metrics_train;
- metrics_eval;
- descriptor;
- score;
- rank;
- artifact paths.

Export top candidates to a C++-loadable catalog:

```text
outputs/search_mvp/<run_name>/catalog.json
outputs/search_mvp/<run_name>/cells/*.f32
outputs/search_mvp/<run_name>/archive.json
outputs/search_mvp/<run_name>/metrics.json
outputs/search_mvp/<run_name>/summary.md
```

The exported catalog must contain `simulation_dims` and `resolution_policy`.

## Tests

Add or update pytest tests for:

- catalog `simulation_dims` and `resolution_policy` export;
- C++-compatible manifest fields;
- hard/ST output range;
- ST gradient nonzero on a short rollout;
- normalized descriptors finite;
- profile losses finite;
- search script smoke run creates catalog and archive.

## Acceptance commands

```powershell
uv run python -m pytest python/tests
```

```powershell
uv run python python/scripts/search_sensorimotor_mvp.py `
  --profile move_shape_target `
  --size 32 `
  --steps 16 `
  --iterations 3 `
  --inner-optim-steps 3 `
  --train-clip-mode hard `
  --out outputs/search_mvp/smoke_move_hard
```

```powershell
uv run python python/scripts/search_sensorimotor_mvp.py `
  --profile maintain_animal_profile `
  --catalog configs/lenia3d_reference/animals.json `
  --animal "Diguttome" `
  --size 64 `
  --steps 16 `
  --iterations 3 `
  --inner-optim-steps 3 `
  --train-clip-mode straight_through_hard `
  --out outputs/search_mvp/smoke_maintain_animal
```

After the second command, the C++ app should be able to open:

```text
outputs/search_mvp/smoke_maintain_animal/catalog.json
```

via the Animal Catalog file picker and render candidates.

## Notes

- Keep implementation small and inspectable.
- Do not introduce obstacles/multi-channel in this plan.
- Do not build full MAP-Elites yet.
- If hard clamp gives weak gradients, do not replace it silently; compare with straight-through and record train/eval modes.
- Use hard-clamp evaluation for all candidates before ranking.
