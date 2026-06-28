# Flow / Expanded model notes for upcoming Plan 08

This document records the main model-layer decisions that should guide the next major implementation after Plan 07.5.

## Core direction

The next model layer should be:

```text
Expanded Lenia + Flow-Lenia-ready + differentiable kernel spec
```

rather than continuing to add search features to the current single-kernel legacy Lenia.

## Minimum next model features

```text
- ModelSpec JSON shared by Python and C++
- multi-channel state
- multi-kernel kernel bank
- additive expanded Lenia update
- Flow-Lenia update mode
- differentiable smooth Gaussian-mixture radial kernels
- legacy Lenia3D -> smooth kernel conversion
- C++ replay of the same ModelSpec
```

## Kernel families

Recommended first implementation:

```text
legacy_shell
smooth_gaussian_mixture
```

Optional schema-only or experimental:

```text
beta_bump
signed_bandpass
```

## Flow transport

Preferred first implementation:

```text
reintegration tracking with uniform cube distribution
```

Special case:

```text
temperature s = 0.5 gives trilinear splat
```

Do not use semi-Lagrangian pullback as the main Flow update because it is not mass-conservative.

## Model search

Do not open the full search space immediately.

First experiments:

```text
A. Expanded additive, C=1, K=2/4
B. Flow, C=1, K=4
C. Flow, C=2, K=4/8
```

Lenia3D should become:

```text
- legacy reference
- seed source
- kernel prior
```

not a hard compatibility constraint.
