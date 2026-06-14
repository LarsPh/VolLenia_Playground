# Review — Plan 02 CUDA volume renderer

Result: no blocking issues found from code review. If local rendering tests pass, proceed to Plan 03.

## What looks good

- The project now builds CUDA renderer files directly into the main target.
- The renderer uses a linear float32 volume and uploads it to a CUDA 3D texture.
- Rendering writes into the existing CUDA/OpenGL PBO path.
- Synthetic presets provide useful debug scenes: sphere, shell, Gaussian blobs, Lenia-like phantom, and axis ramp.
- Render modes include emission-absorption, MIP, and first-hit, which is enough for debugging volume coordinates and thresholds.
- The camera orbit/zoom path is sufficient for simulation review.

## Non-blocking notes

- `render.target_width` / `target_height` in config are not used yet. Keep this as a future render-scale feature.
- The app is currently VSync-limited via `glfwSwapInterval(1)`. Add a VSync checkbox later when CUDA event timings are added.
- `CudaVolumeRenderer::setVolume(const SyntheticVolume&)` should be refactored in Plan 03 to accept a generic device volume view.
- `setVolume()` currently recreates the CUDA 3D array/texture. Plan 03 should reuse them when dimensions are unchanged and only copy new data each frame.
- PBO -> texture upload and single PBO are fine for now; optimize only after profiling.

## Recommended next step

Proceed with Plan 03: single-channel 3D Lenia simulation using CUDA + cuFFT.
