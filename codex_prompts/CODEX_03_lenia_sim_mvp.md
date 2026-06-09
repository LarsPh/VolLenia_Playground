# Codex prompt — 03 Single-channel 3D Lenia simulation MVP

You are working in `D:\projects\VolLenia_Playground`.

The repo should already have a CUDA volume renderer that can render synthetic volumes.

## Goal

Implement a single-channel 3D Lenia simulation using CUDA + cuFFT, and render the current state using the existing CUDA volume renderer.

## Local reference

Use `D:\projects\Lenia3D` only as an algorithm reference. Important files:

```text
src/core/LeniaEngine.js
src/core/KernelGen.js
src/core/FftUtils.js
```

Core algorithm:

```text
A -> FFT
Ahat * Khat -> Uhat
IFFT -> U
growth(U, mu, sigma)
A = clamp(A + dt * growth, 0, 1)
```

## Constraints

- Do not import TensorFlow.js or WebGL code.
- Do not implement multi-channel yet.
- Do not implement animals3D importer yet.
- Do not introduce OptiX/Falcor/pbrt.
- No CPU readback of full volume.
- Renderer must remain generic; add a packer if needed.

## Suggested files

```text
src/core/VolumeDesc.h
src/sim/DeviceVolume.h
src/sim/DeviceVolume.cu
src/sim/Rule.h
src/sim/LeniaRule.h
src/sim/LeniaRule.cu
src/sim/CufftContext.h
src/sim/CufftContext.cpp
src/sim/KernelGenerator.h
src/sim/KernelGenerator.cu
src/sim/GrowthFunctions.h
src/sim/GrowthFunctions.cu
src/render/RenderVolumePacker.h
src/render/RenderVolumePacker.cu
configs/lenia.default.json
src/app/UiPanel.cpp
CMakeLists.txt
```

## FFT plan

Use real-to-complex and complex-to-real 3D cuFFT if practical:

```text
A:    float [nx, ny, nz]
Ahat: complex [nx, ny, nz/2+1]
Khat: complex [nx, ny, nz/2+1]
Uhat: complex [nx, ny, nz/2+1]
U:    float [nx, ny, nz]
```

Remember to normalize inverse FFT by `1.0f / (nx * ny * nz)`.

## Kernel MVP

Implement a radial shell kernel:

```text
r = distance / radius
if r < 1:
  shell_index = floor(r * B)
  local_r = fract(r * B)
  weight = shell_weights[shell_index] * core(local_r)
else:
  weight = 0
normalize sum to 1
roll center to origin
FFT to Khat
```

Start with polynomial core `(4*r*(1-r))^4`.

## Growth MVP

```cpp
G(U) = 2 * exp(-(U - mu)^2 / (2 * sigma * sigma)) - 1;
A = clamp(A + dt * G(U), 0, 1);
```

## UI controls

- play/pause
- single step
- reset random seed
- steps per frame
- mu
- sigma
- dt or T
- kernel radius
- shell weight presets

## Acceptance criteria

- `128^3` Lenia state can step continuously.
- Renderer displays the Lenia state.
- Play/pause/step/reset works.
- Changing kernel params rebuilds kernel.
- No NaN/Inf after stepping.
- No full-volume CPU readback.

After implementation, summarize changes and provide build/run commands.
