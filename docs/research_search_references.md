# Research references for VolLenia differentiable search

This document collects the near-term reading list and explains why each reference matters for the PyTorch differentiable search direction.

## Core direction

VolLenia is moving toward a dual-backend architecture:

```text
PyTorch backend:
  differentiable simulation, short-horizon training, metrics, objectives, search

C++/CUDA backend:
  high-resolution replay, volume rendering, interactive inspection, video capture
```

The research goal is not just a 3D Lenia preset zoo. The target is:

```text
3D self-organizing volumetric agents
+ environment coupling
+ differentiable search
+ robust metrics
+ high-quality C++ visualization
```

## Must-read references

### Sensorimotor Lenia

- Paper: Discovering Sensorimotor Agency in Cellular Automata using Diversity Search
- Site / code: https://developmentalsystems.org/sensorimotor-lenia/
- GitHub: https://github.com/flowersteam/sensorimotor-lenia-search
- arXiv: https://arxiv.org/abs/2402.10236

Why it matters:

```text
This is the closest methodological ancestor.
It combines diversity search, curriculum learning, and gradient descent to find CA agents that move, maintain body integrity, and react coherently to obstacles.
```

Design ideas to borrow:

```text
outer diversity / IMGEP-style exploration
inner differentiable rollout + gradient descent
obstacle/environment channels
goal-conditioned losses
robustness/generalization tests
scale perturbation tests
```

### Leniabreeder / Quality-Diversity Lenia

- Paper: Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity
- arXiv: https://arxiv.org/abs/2406.04235

Why it matters:

```text
This frames Lenia discovery as Quality-Diversity search instead of hand-tuned preset hunting.
It explicitly targets diverse autonomous patterns and can use manual or unsupervised diversity measures.
```

Design ideas to borrow later:

```text
MAP-Elites-like archive
behavior descriptors
fitness + diversity separation
manual descriptors first, learned descriptors later
not one best organism, but a repertoire of niches
```

### Growing Neural Cellular Automata

- Distill: https://distill.pub/2020/growing-ca/

Why it matters:

```text
Shows how differentiable local rules can learn stable morphogenesis.
Sample-pool and damage-training ideas are useful for robustness, even if VolLenia does not become a pure NCA.
```

Design ideas to borrow later:

```text
sample pool
damage pool
self-repair as robustness, not main contribution
hidden channels once we move toward Neural/Flow VolLenia
```

### Cells2Pixels

- Project: https://cells2pixels.github.io/
- arXiv: https://arxiv.org/abs/2506.22899

Why it matters:

```text
A modern NCA graphics route that separates coarse self-organizing dynamics from high-resolution rendering via local implicit decoding.
VolLenia should not clone it, but it is useful context for graphics-facing positioning.
```

### Flow-Lenia

- arXiv early version: https://arxiv.org/abs/2212.07906
- arXiv newer version: https://arxiv.org/abs/2506.08569
- Project page: https://sites.google.com/view/flowlenia/

Why it matters:

```text
Mass conservation and parameter localization are likely the long-term route for environment/resource-driven VolLenia.
```

Near-term use:

```text
Do not implement Flow in Plan 06.
Use it as motivation for future resource/transport objectives.
```

### Exploring Flow-Lenia Universes with a curiosity-driven AI scientist

- arXiv: https://arxiv.org/abs/2505.15998

Why it matters:

```text
Shows a system-level search direction for Flow-Lenia using curiosity-driven exploration and simulation-wide metrics such as evolutionary activity, compression-based complexity, and multi-scale entropy.
This is useful later when VolLenia moves from individual agent search to ecosystem-level search.
```

### Adaptive Exploration in Lenia with Intrinsic Multi-Objective Ranking

- arXiv: https://arxiv.org/abs/2506.02990

Why it matters:

```text
Useful later for intrinsic objectives:
distinctiveness, population sparsity, homeostatic regulation.
```

### PyTorch FFT with autograd

- PyTorch blog: https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/

Why it matters:

```text
The PyTorch backend can use torch.fft for GPU-accelerated differentiable convolution. No custom CUDA backward is needed for Plan 06.
```

## Metrics vocabulary

Plan 06 should implement primitive metrics, not final scoring.

### Core body metrics

```text
mass
mean_density
max_density
active_voxels
center_of_mass
second_moment
covariance
anisotropy
border_mass
```

### Environment metrics

```text
target_distance
obstacle_overlap
resource_overlap
damage_recovery later
```

### Motion metrics

```text
COM displacement
speed in body units
trajectory smoothness
body-frame displacement later
```

### Stability metrics

```text
mass mean/variance over rollout
compactness mean/variance
anisotropy mean/variance
NaN/Inf flag
```

## Search vocabulary

### Random search

Good for sanity and early baseline. Bad for high-dimensional discovery.

### Local mutation

Useful around known animals or known PyTorch candidates.

### MAP-Elites / Quality-Diversity

Recommended after metrics are stable. Separate:

```text
quality:
  survive, localized, coherent, non-exploding

descriptor:
  speed, body size, anisotropy/blobness, oscillation, resource seeking
```

### IMGEP / curiosity-driven search

Recommended later when moving from fixed descriptors to goal-conditioned exploration.

### Inner gradient refinement

Differentiable local optimizer. Best used as:

```text
outer diversity search proposes candidate
inner gradient search improves candidate for a sampled goal/task
```

## Near-term principle

Do not optimize a giant scalar reward first.

Plan 06 should build metrics and toy objectives. Plan 07 can begin task losses. Plan 08/09 can add QD/IMGEP.
