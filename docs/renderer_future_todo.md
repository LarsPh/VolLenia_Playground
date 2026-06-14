# Renderer / GPU interop future TODO

This document collects rendering and CUDA/OpenGL interop ideas that are useful later but should not block the current simulation milestones.

## Current baseline

The current renderer path is intentionally simple:

```text
CUDA simulation/render kernel writes uchar4 image -> OpenGL PBO
OpenGL uploads PBO to a 2D texture -> fullscreen triangle
ImGui overlays controls
```

For the next milestone, keep this path. It is good enough for debugging Lenia dynamics and avoids mixing simulation correctness with graphics-pipeline experiments.

## Near-term renderer polish

### Gradient-based local lighting

Add a mode that estimates the volume gradient by sampling neighboring density values:

```text
grad = [rho(x+dx)-rho(x-dx), rho(y+dy)-rho(y-dy), rho(z+dz)-rho(z-dz)]
normal = normalize(-grad)
```

Use it for cheap diffuse/rim lighting in emission-absorption mode. This is not physically correct volume scattering, but it gives much better shape cues for Lenia bodies.

Recommended controls:

```text
Enable gradient lighting
Light direction
Diffuse strength
Rim strength
Gradient sample scale
```

Do not do real light integration yet. True single/multiple scattering quickly becomes path tracing-like: more expensive, more noisy, and harder to debug while simulation is still evolving.

### Transfer function editor

Replace the hardcoded blue/cyan/orange mapping with a small editable transfer function:

```text
density -> color
density -> alpha multiplier
```

A minimal first version can be 3 or 4 color stops exposed in ImGui.

### Debug render modes

Keep and extend non-photoreal debug modes:

```text
Emission-absorption
MIP
First hit / threshold surface
Slice view: X/Y/Z plane
Depth / accumulated alpha view
```

These are useful when validating texture coordinates, Lenia state ranges, and sparse/dense behavior.

## Performance / interop optimization backlog

### Cache OpenGL uniform locations

`GlDisplay` currently may query uniforms during draw. This is negligible compared with ray marching and cuFFT, but can be cleaned up when the display pipeline gets more uniforms.

### Optional render target size

`configs/app.default.json` has `render.target_width` and `render.target_height`, but the current renderer uses the window framebuffer size. That is fine while performance is high.

Later, add:

```text
Render scale: 0.25x / 0.5x / 0.75x / 1.0x
Fixed render target size override
Upscale to framebuffer
```

This will become useful when simulation + volume rendering + gradient lighting exceed the display refresh budget.

### VSync / frame-limit controls

Add a UI checkbox or config flag:

```text
VSync on/off -> glfwSwapInterval(1/0)
Optional frame limiter
Show raw GPU timings
```

This is useful for profiling, but not necessary before cuFFT simulation is in place. Once cuFFT is added, use CUDA events around simulation and render kernels instead of relying only on FPS.

### Double PBO / ring-buffer output

If profiling shows `cudaGraphicsMapResources` / `cudaGraphicsUnmapResources` or `glTexSubImage2D` causes stalls, introduce two or three PBOs:

```text
frame N: CUDA writes PBO[i]
frame N: OpenGL displays PBO[i-1]
```

Only do this after measuring a stall. It adds lifetime/synchronization complexity and is not needed for the current MVP.

### Direct CUDA write to OpenGL texture/surface

Alternative output path:

```text
Register GL_TEXTURE_2D with CUDA
Map to CUDA array
Write via surface object
Display texture directly
```

This removes the PBO -> texture upload, but it couples output storage more tightly to GL texture interop. Keep PBO output until the renderer becomes a bottleneck.

## Sparse volume acceleration backlog

### Adaptive step size

For low-density space, increase step size; near high density or high gradient, reduce step size. Start simple:

```text
if sample < threshold: t += empty_step
else: t += fine_step
```

This can help Lenia because many states are localized in a mostly empty box.

### Empty-space skipping with brick min/max

Build a coarse brick grid, for example 8^3 or 16^3 bricks. Store per-brick:

```text
min density
max density
optional occupancy flag
```

During ray marching, skip entire bricks whose max density is below threshold. This is a good medium-term optimization once Lenia patterns are sparse and stable.

### Sparse volume structures / NanoVDB

NanoVDB/OpenVDB-style sparse volumes are interesting later, especially for exporting/importing volumes or rendering very large sparse worlds. Do not introduce them in the near term:

```text
Current Lenia state is dense and changes every simulation step.
cuFFT convolution naturally wants dense linear buffers.
Renderer already has enough work before sparse structures.
```

A practical future use is exporting a frozen Lenia frame to NanoVDB for cinematic rendering, not driving the primary simulation state.

## Physically based rendering backlog

Potential later stages:

```text
Single scattering shadow ray
Directional light through volume
Temporal accumulation
Denoising
OptiX backend for volume + mesh scenes
Offline pbrt export for stills
```

Keep these as late-stage visual/cinematic features. The near-term goal is still simulation exploration, not a path tracer.

## Priority summary

Recommended order after Lenia simulation is running:

```text
1. CUDA event timings for simulation/upload/render
2. Gradient lighting
3. Transfer function editor
4. Render scale / target resolution
5. VSync toggle and profiling controls
6. Empty-space skipping / brick min-max
7. Double PBO only if measured stalls exist
8. Direct GL texture/surface output only if PBO path is a bottleneck
9. NanoVDB / OptiX / path-tracing features much later
```
