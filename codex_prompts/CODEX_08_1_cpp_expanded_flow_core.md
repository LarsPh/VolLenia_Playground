# CODEX 08 Stage 1 — Implement C++ Expanded/Flow Model Core

You are working in `D:\projects\VolLenia_Playground`.

Implement the C++ portion of PLAN 08 Stage 1. Do not rewrite Python search in this milestone.

Read first:

```text
docs/plan08_stage1_algorithm_design.md
plans/PLAN_08_STAGE1_cpp_expanded_flow_core.md
```

Also inspect current code patterns:

```text
src/sim/LeniaSimulation.*
src/sim/DeviceVolume.*
src/render/CudaVolumeRenderer.*
src/io/LeniaAnimalCatalog.*
src/app/App.*
src/app/UiPanel.*
```

If available, also inspect:

```text
D:\projects\FlowLenia\flowlenia\flowlenia.py
D:\projects\FlowLenia\flowlenia\reintegration_tracking.py
D:\projects\FlowLenia\flowlenia\utils.py
```

Reference behavior:

- FlowLenia uses multi-kernel FFT, growth, channel aggregation, Sobel gradients, and reintegration tracking.
- Implement the C++ version cleanly and with performance awareness.

Hard requirements:

1. Do not break existing legacy Lenia3D animal catalog mode.
2. Add new model code in new files rather than bloating `LeniaSimulation`.
3. Use channel-major device layout.
4. Use FFT for multi-kernel convolution.
5. Use 3D Sobel gradient in Flow mode.
6. Use target-centric reintegration tracking with fixed `transport_sigma = 0.5`.
7. Do not implement parameter localization yet.
8. Do not implement resource/obstacle channels yet.
9. Provide minimal ModelSpec presets under `configs/modelspec/`.
10. Build Release successfully.

Suggested file additions:

```text
src/model/ModelSpec.h
src/model/ModelSpec.cpp
src/model/KernelSpec.h
src/sim/DeviceMultiVolume.h
src/sim/DeviceMultiVolume.cu
src/sim/KernelBank.h
src/sim/KernelBank.cu
src/sim/ExpandedFlowSimulation.h
src/sim/ExpandedFlowSimulation.cu
src/sim/FlowTransport.h
src/sim/FlowTransport.cu
configs/modelspec/*.json
```

Implementation details:

- Smooth Gaussian mixture kernel:
  `q = radius/R`, `envelope = sigmoid(k*(1-q))`, sum Gaussian bumps, normalize sum to 1.
- Expanded additive:
  `U[dst] += h * G(IFFT(FFT(A[src]) * FFT(K)))`.
- Flow:
  `F = (1-alpha)*grad_U - alpha*grad_A_sum`.
- Reintegration:
  target-centric gather over `[-dd,dd]^3`; `dd = ceil(flow_max + sigma - 0.5 + eps)`, sigma fixed 0.5.

Testing / acceptance:

- Add a visible `ModelSpec` mode to UI or a minimal command path to load a preset.
- Show selected render channel.
- Output kernel profile CSV and mid-slice `.f32` for at least one kernel.
- Log mass before/after Flow steps for sanity.
- Keep CMake clean.

Avoid overengineering:

- No Python parity yet.
- No search integration yet.
- No OpenES yet.
- No custom CUDA atomics source-splat backend yet.
