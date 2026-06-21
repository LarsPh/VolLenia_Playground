# Sensorimotor-style search design notes for VolLenia Plan 07

## Why this plan is not only a cleanup patch

The next step should not be a list of isolated fixes. Sensorimotor Lenia shows that the search algorithm, clamp semantics, loss, metrics, archive, and environment randomization are coupled design choices.

For Plan 07, VolLenia intentionally avoids obstacles/resources and focuses on two core questions:

1. Can a VolLenia candidate be optimized toward a target position while keeping a coherent body?
2. Can we use the same differentiable loop to expand the stable neighborhood around Lenia3D animals, including unstable/rescale-fragile animals?

## Sensorimotor pattern being adapted

The reference search pattern is:

```text
random policy initialization
  -> rollout
  -> map observation to goal/behavior descriptor
  -> archive policy + reached goal
  -> sample new target goal
  -> select archived policy closest to target
  -> copy/mutate source policy
  -> BPTT/Adam toward target
  -> evaluate across multiple randomized rollouts
  -> archive improved policy
```

VolLenia Plan 07 adapts this but keeps the first pass simpler:

```text
no obstacles
single-channel 3D Lenia
small archive
hard/ST clamp comparison
C++ catalog export for visual inspection
```

## Clamp choice

- `hard` is closest to C++ and original Lenia.
- `straight_through_hard` uses hard forward dynamics but a biased surrogate gradient.
- `none` is useful only for debugging gradients and should not be a search default.
- `soft_tanh` is a future surrogate/ablation because it changes the dynamics.

Plan 07 should train with `hard` or `straight_through_hard`, but always evaluate and rank with `hard`.

## Metrics should be normalized

Search should not hard-code absolute mass or distance thresholds that only make sense at one resolution. Prefer normalized descriptors:

```text
mass_fraction = mass / volume_voxels
active_fraction = active_voxels / volume_voxels
com_norm = COM / (shape - 1)
second_moment_norm = second_moment / min_dim^2
body_radius = sqrt(second_moment)
target_distance_body = target_distance / body_radius
target_distance_norm = target_distance / min_dim
mass_ratio = mass / initial_mass
active_ratio = active_voxels / initial_active_voxels
compactness_ratio = second_moment / initial_second_moment
anisotropy = max covariance eig / min covariance eig
border_mass = border density fraction
```

These let us compare 32³, 64³, and later 128³ candidates more meaningfully.

## Profiles

### `move_shape_target`

Use a target sphere or COM target. It tests whether BPTT can push an organism-like density toward a goal without letting it vanish, explode, become a cylinder, or hit borders.

### `maintain_animal_profile`

Use initial animal metrics as reference. This is the direct bridge to the Lenia3D catalog. It asks whether local variants remain alive and visually coherent.

### `rescue_unstable_animal`

Start from manually observed unstable/rescale-fragile Lenia3D animals. Use Profile B's stability objective plus mutation/optimization to find nearby stable variants.

## Why no obstacle this time

Obstacles are a core Sensorimotor Lenia idea, but adding them to VolLenia means introducing environment channels or special collision fields. That is the next curriculum stage. Plan 07 focuses on the search infrastructure and goal/loss semantics first.

## Lenia3D resolution semantics

Lenia3D's default simulation dimension is `[64,64,64]`. Its animal RLE cells are cropped organisms loaded into this 64³ world. VolLenia catalogs should represent that explicitly:

```json
"dims": [cropped_nx, cropped_ny, cropped_nz],
"simulation_dims": [64,64,64],
"resolution_policy": "cropped"
```

PyTorch full-state exports should use:

```json
"dims": [nx,ny,nz],
"simulation_dims": [nx,ny,nz],
"resolution_policy": "native"
```
