# Simulation future TODO

This document collects simulation-side ideas that should not block PLAN 04.

## Current baseline

```text
single-channel 3D Lenia
float32 dense grid
cuFFT R2C/C2R convolution
single radial kernel
one growth function at a time
renderer consumes DeviceVolumeView
```

## Near-term

### CUDA event timings

Add timing around:

```text
cuFFT R2C
spectrum multiply
cuFFT C2R
growth/update kernel
renderer upload cudaMemcpy3D
raymarch kernel
PBO map/unmap/display path
```

Do this before optimizing PBOs or renderer output paths.

### Optional validation mode

Current simulation validation can synchronize by copying an invalid flag back to CPU. Keep it for debugging, but later add:

```text
[ ] Validate NaN/Inf every step
```

Default off when exploring performance.

### Higher grid sizes

Expose more grid options once PLAN 04 is stable:

```text
64, 96, 128, 160, 192, 256
```

Possible later options:

```text
320, 384, 512
```

Those will likely need profiling, memory budgeting, and possibly lower render resolution.

## Mid-term

### Better species/preset search

After imported animal presets work:

```text
save current edited preset
randomize shell weights around current animal
small parameter mutation UI
batch screenshot/recording
basic survival/mass statistics
```

### Multi-kernel / multi-channel

The next major simulation expansion should be designed as:

```text
basis kernels K_k
channels A_c
potentials U_{c,k} = K_k * A_c
mixing weights into growth/update
```

Do not bolt this onto the single-channel path too early.

### Environment fields

Keep environment/resource fields as a separate branch after the biology catalog works:

```text
nutrient
poison/obstacle
metabolic cost
growth depends on local resource
```

## Long-term

```text
Flow-Lenia-like conservative transport
spectral/logit-field Lenia
learnable kernels/growth functions
quality-diversity search
multi-species/rule-code channels
```
