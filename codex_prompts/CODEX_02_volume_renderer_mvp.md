# Codex prompt — 02 CUDA volume renderer MVP

You are working in `D:\projects\VolLenia_Playground`.

The repo should already have a GLFW/ImGui app and CUDA/OpenGL PBO smoke test.

## Goal

Implement a minimal CUDA volume renderer that ray marches a synthetic 3D float volume and writes the image to the existing OpenGL PBO.

## Local reference

Use the local CUDA sample only as a reference:

```text
D:\projects\cuda_samples_13.3
```

Search for:

```text
volumeRender.cpp
volumeRender_kernel.cu
```

You may reference the ray-box intersection / texture sampling / alpha compositing ideas. If copying code, preserve original copyright/license headers.

## Constraints

- Do not implement Lenia yet.
- Do not use cuFFT yet.
- Do not introduce OptiX/Falcor/pbrt/Vulkan/DirectX.
- Do not read volume or image back to CPU.
- Keep renderer reusable; it should consume a generic volume, not a Lenia-specific class.

## Suggested files

```text
src/render/CudaVolumeRenderer.h
src/render/CudaVolumeRenderer.cu
src/render/RenderParams.h
src/render/TransferFunction.h
src/render/TransferFunction.cpp
src/render/SyntheticVolume.h
src/render/SyntheticVolume.cu
src/app/Camera.h
src/app/Camera.cpp
src/app/UiPanel.cpp
CMakeLists.txt
```

## Features

- Allocate a synthetic `float32` volume on GPU, default `128^3`.
- Copy/pack it into a CUDA 3D array.
- Create a CUDA texture object with normalized coords, clamp addressing, linear filtering.
- Implement ray-box intersection.
- Implement fixed-step ray marching.
- Implement emission-absorption front-to-back compositing:

```cpp
alpha = 1.0f - expf(-density_scale * density * step_size);
accum.rgb += transmittance * alpha * color;
transmittance *= (1.0f - alpha);
```

- Add early exit when transmittance is low.
- Render into existing PBO.

## Synthetic volume presets

Implement at least:

- sphere
- shell
- Gaussian blobs

## UI controls

- preset
- step size
- density scale
- threshold
- brightness/exposure
- max steps
- reset camera

## Acceptance criteria

- App shows a semi-transparent 3D volume.
- UI changes affect rendering live.
- Orbit/zoom camera works.
- Resize works.
- No CPU readback.
- Renderer code is not tied to Lenia.

After implementation, summarize changes and provide build/run commands.
