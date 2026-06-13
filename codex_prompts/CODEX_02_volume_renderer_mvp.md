# Codex prompt — 02 CUDA volume renderer MVP

You are working in `D:\projects\VolLenia_Playground`.

The repo should already have a GLFW/ImGui app and CUDA/OpenGL PBO smoke test.

## Goal

Implement a minimal CUDA volume renderer that ray marches a synthetic 3D float volume and writes the image to the existing OpenGL PBO.

The intended GPU path is:

```text
synthetic CUDA linear volume -> CUDA 3D array/texture object -> CUDA ray marcher -> existing PBO -> existing OpenGL display
```

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

You may reference ray-box intersection, `tex3D<float>()`, transfer functions, front-to-back compositing, and opacity early exit. If copying code, preserve original copyright/license headers. Prefer rewriting in the current project style.

## Constraints

- Do not implement Lenia yet.
- Do not use cuFFT yet.
- Do not introduce OptiX/Falcor/pbrt/Vulkan/DirectX.
- Do not read volume or image back to CPU.
- Keep renderer reusable; it should consume a generic volume, not a Lenia-specific class.
- Reuse the existing Plan 1 PBO output and `GlDisplay`.
- Do not replace PBO output with direct CUDA writes to a GL texture in this milestone.
- Do not implement double PBO/ring buffering unless the current single PBO path is broken.
- Avoid every-frame resource rebuilds. Rebuild CUDA 3D array / texture object only when volume dimensions or preset changes.

## Suggested files

```text
src/render/CudaVolumeRenderer.h
src/render/CudaVolumeRenderer.cu
src/render/RenderParams.h
src/render/SyntheticVolume.h
src/render/SyntheticVolume.cu
src/app/Camera.h
src/app/Camera.cpp
src/app/App.h
src/app/App.cpp
src/app/UiPanel.cpp
CMakeLists.txt
configs/app.default.json
```

`TransferFunction.*` is optional for this milestone. A simple in-kernel transfer function is acceptable.

## Required features

- Allocate a synthetic `float32` volume on GPU, default `128^3`.
- Generate synthetic presets entirely on GPU.
- Copy/pack the linear volume into a CUDA 3D array.
- Create a CUDA texture object:
  - normalized coordinates
  - clamp addressing
  - linear filtering
  - float read mode
- Implement ray-box intersection for a `[-1, 1]^3` volume box.
- Generate perspective camera rays from explicit camera vectors, not OpenGL matrix state.
- Implement fixed-step ray marching.
- Implement emission-absorption front-to-back compositing:

```cpp
alpha = 1.0f - expf(-density_scale * density * step_size);
accum.rgb += transmittance * alpha * color;
transmittance *= (1.0f - alpha);
```

- Add early exit when transmittance is low.
- Render into the existing PBO.

## Synthetic volume presets

Implement at least:

- sphere density
- shell density
- Gaussian blobs
- Lenia-like phantom: shell + internal blobs + wispy bands

Optional debug presets:

- x/y/z ramp
- stripes/checker
- noise blob

## Render modes

Implement `EmissionAbsorption` as the main mode.

Also implement at least one debug mode:

- `MIP`: max intensity projection
- `FirstHit`: threshold/iso-surface-ish first hit

Debug mode is useful to verify camera orientation and texture coordinates.

## Camera controls

Add a real interactive camera:

- left mouse drag: orbit yaw/pitch
- mouse wheel: zoom/distance
- reset camera button

Pan is optional. Avoid coupling renderer to OpenGL matrices. Pass camera position/forward/right/up/fov/aspect into CUDA.

## UI controls

Add controls for:

- volume preset
- volume resolution
- regenerate volume button
- step size
- density scale
- threshold
- brightness/exposure
- max steps
- render mode
- camera FOV/distance
- reset camera

Add status/debug text:

- volume dimensions
- render target dimensions
- CUDA texture/volume status
- last render error

## Acceptance criteria

- App shows a semi-transparent 3D volume.
- Sphere, shell, Gaussian blobs, and Lenia-like phantom are visible and distinguishable.
- UI changes affect rendering live.
- Orbit/zoom camera works.
- MIP or FirstHit debug mode works.
- Resize works.
- No CPU readback.
- Renderer code is not tied to Lenia.
- Shutdown is clean: no CUDA array/texture/PBO resource leak or crash.

After implementation, summarize changes and provide build/run commands.
