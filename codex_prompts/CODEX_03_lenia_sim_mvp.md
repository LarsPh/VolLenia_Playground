# Codex prompt — 03 Single-channel 3D Lenia simulation MVP

You are working in `D:\projects\VolLenia_Playground`.

The repo already has:

- GLFW / ImGui app shell.
- CUDA/OpenGL PBO output path.
- CUDA volume renderer that ray marches synthetic float32 volumes.

## Goal

Implement a single-channel 3D Lenia simulation using CUDA + cuFFT and render the current Lenia state with the existing CUDA volume renderer.

The most important parts are correct FFT layout, correct kernel normalization, correct inverse FFT scaling, and keeping the renderer generic.

## Local references

Use `D:\projects\Lenia3D` only as algorithm/parameter reference. Important files:

```text
src/core/LeniaEngine.js
src/core/KernelGen.js
src/core/FftUtils.js
src/data/animals3D.js
```

Core algorithm:

```text
A -> R2C FFT
Ahat * Khat -> Uhat
C2R IFFT -> U
U *= 1 / (nx * ny * nz)
growth = G(U; mu, sigma)
A = clamp(A + (1/T) * growth, 0, 1)
```

Do not import TensorFlow.js or WebGL code.

## Constraints

- Do not implement multi-channel yet.
- Do not implement the animals3D RLE importer yet.
- Do not introduce OptiX, Falcor, pbrt, Vulkan, or DirectX.
- Do not read back the full simulation volume to CPU.
- Do not store the primary simulation state in a CUDA array; keep it as a linear `float*` for cuFFT.
- Renderer must remain generic. It should consume a generic device volume view, not a Lenia-specific class.
- Keep synthetic volume rendering available as a debug/source option.

## Required CMake change

Link cuFFT:

```cmake
target_link_libraries(VolLenia_Playground PRIVATE
    CUDA::cudart
    CUDA::cufft
    ...
)
```

Add all new `.cu/.cpp` files to `add_executable(...)`.

## Suggested files

```text
src/core/VolumeDesc.h
src/sim/CufftCheck.h
src/sim/DeviceVolume.h
src/sim/DeviceVolume.cu
src/sim/LeniaParams.h
src/sim/LeniaSimulation.h
src/sim/LeniaSimulation.cu
src/sim/LeniaSeeds.h
src/sim/LeniaSeeds.cu
src/render/CudaVolumeRenderer.h
src/render/CudaVolumeRenderer.cu
src/render/SyntheticVolume.h
src/render/SyntheticVolume.cu
src/app/App.h
src/app/App.cpp
src/app/UiPanel.h
src/app/UiPanel.cpp
configs/lenia.default.json
CMakeLists.txt
```

It is okay to choose slightly different names, but keep the architecture modular.

## Required core refactor

Move or define a shared volume view:

```cpp
struct VolumeDesc {
    int nx = 128;
    int ny = 128;
    int nz = 128;
};

struct DeviceVolumeView {
    VolumeDesc desc;
    const float* data = nullptr;
};
```

Current linear layout must remain:

```cpp
index = (z * ny + y) * nx + x; // x fastest
```

## Required renderer refactor

Refactor the renderer from `setVolume(const SyntheticVolume&)` to a generic upload function, for example:

```cpp
void uploadVolume(DeviceVolumeView volume);
```

Requirements:

- If the volume dimensions changed, recreate the CUDA 3D array and texture object.
- If dimensions are unchanged, reuse the existing CUDA 3D array and texture object and only do `cudaMemcpy3D` device-to-device.
- `CudaVolumeRenderer` must not include `SyntheticVolume.h`.
- `SyntheticVolume` should expose `DeviceVolumeView view() const`.
- `LeniaSimulation` should expose `DeviceVolumeView currentStateView() const`.

## cuFFT layout

Because the existing volume layout is x-fastest, create 3D cuFFT plans as:

```cpp
cufftPlan3d(&r2c_plan, nz, ny, nx, CUFFT_R2C);
cufftPlan3d(&c2r_plan, nz, ny, nx, CUFFT_C2R);
```

Use out-of-place buffers for the MVP:

```text
A:     float         [nx * ny * nz]
U:     float         [nx * ny * nz]
Ahat:  cufftComplex  [nz * ny * (nx/2 + 1)]
Khat:  cufftComplex  [nz * ny * (nx/2 + 1)]
Uhat:  cufftComplex  [nz * ny * (nx/2 + 1)]
```

After C2R, normalize:

```cpp
u = U[i] * (1.0f / float(nx * ny * nz));
```

## Simulation step

Implement:

```cpp
cufftExecR2C(r2c_plan, A, Ahat);
launchMultiplySpectrum(Uhat, Ahat, Khat, spectrum_count);
cufftExecC2R(c2r_plan, Uhat, U);
launchLeniaUpdate(A, U, voxel_count, inv_n, mu, sigma, T);
```

Growth:

```cpp
float g = 2.0f * expf(-((u - mu) * (u - mu)) / (2.0f * sigma * sigma)) - 1.0f;
float dt = 1.0f / T;
A[i] = clamp(A[i] + dt * g, 0.0f, 1.0f);
```

Numerical guards:

- `sigma >= 1e-5f`
- `T >= 1.0f`
- If a result is NaN/Inf, avoid propagating it and report an error/status if practical.

## Kernel generation

Implement a Lenia3D-style radial shell kernel. MVP may generate the spatial kernel on CPU for simplicity/correctness, then upload to GPU and compute `Khat` with cuFFT. Kernel rebuilds are infrequent, so this is acceptable for Plan 03.

Kernel parameters:

```cpp
radius R in voxels
shell weights b[]
core = polynomial by default
```

Polynomial core:

```cpp
core(r) = pow(4*r*(1-r), 4), for r in [0,1]
```

Shell mapping:

```text
q = distance / R
if q >= 1: K = 0
else:
  s = q * B
  shell_index = clamp(floor(s), 0, B-1)
  local_r = fract(s)
  K = b[shell_index] * core(local_r)
```

Normalize with double precision on CPU:

```cpp
sum = double sum of all K values
K[i] /= sum
```

If `sum <= 0`, throw an error.

For circular convolution alignment, prefer direct origin-periodic distance:

```cpp
dx = min(x, nx - x);
dy = min(y, ny - y);
dz = min(z, nz - z);
distance = sqrt(dx*dx + dy*dy + dz*dz);
```

This avoids an explicit roll and places the kernel center at FFT origin.

## Parameter presets

Add a few Lenia3D-inspired presets. They do not need to reproduce exact animals yet because RLE import is Plan 04.

```json
{
  "Diguttome saliens": {"R": 10, "T": 10, "b": [1.0, 0.75, 0.5833333, 0.9166667], "mu": 0.12, "sigma": 0.01},
  "Diguttome tardus":  {"R": 10, "T": 10, "b": [0.6666667, 1.0, 0.8333333], "mu": 0.15, "sigma": 0.016},
  "Triguttome labens": {"R": 10, "T": 10, "b": [1.0, 0.4166667, 0.0833333, 0.1666667], "mu": 0.16, "sigma": 0.015}
}
```

## Initial state presets

Implement procedural seeds, generated on GPU or CPU-uploaded:

- Centered random ball.
- Asymmetric Gaussian cluster.
- Shell + internal blobs.
- Small impulse/blob for convolution sanity checks.

Use deterministic hash-based random numbers; do not add cuRAND for this milestone.

## UI controls

Keep synthetic volume rendering as a debug source. Add a source selector:

```text
Source: Synthetic / Lenia
```

Lenia controls:

- Play/Pause
- Single step
- Reset seed
- Regenerate seed
- Steps per frame
- Seed preset
- Parameter preset
- R
- T
- mu
- sigma
- Rebuild kernel button
- Generation count
- Simulation/kernel status

Changing `mu`, `sigma`, or `T` should not rebuild the cuFFT plans. Changing volume size or kernel radius/shell weights should rebuild the kernel; changing volume size should recreate plans and buffers.

## Acceptance criteria

- 64^3, 96^3, and 128^3 Lenia state can initialize and step.
- 128^3 can run continuously and render through the existing volume renderer.
- Synthetic / Lenia source toggle works.
- Play/Pause/Step/Reset works.
- Changing kernel parameters rebuilds the kernel.
- Changing growth parameters affects dynamics without rebuilding FFT plans.
- Inverse FFT is normalized by `1/(nx*ny*nz)`.
- State remains in [0,1] and does not produce NaN/Inf.
- No full-volume CPU readback.
- Renderer remains generic and is not tied to LeniaSimulation.

After implementation, summarize changes and provide build/run commands.
