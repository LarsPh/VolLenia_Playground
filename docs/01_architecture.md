# 01 — 架构设计

## 推荐技术栈

```text
Language:
  C++20
  CUDA C++

Build:
  CMake
  CUDAToolkit package

Simulation:
  CUDA Runtime
  cuFFT
  custom CUDA kernels

Rendering:
  CUDA ray marcher
  CUDA 3D texture object / cudaArray
  OpenGL PBO interop
  OpenGL fullscreen display

Window/UI:
  GLFW
  Dear ImGui
  optional ImPlot

Config/Data:
  JSON, e.g. nlohmann/json
```

## 第一阶段 GPU 数据流

```text
SimState.A[0] float32 linear CUDA buffer
        |
        | after simulation step or when rendering requested
        v
RenderVolumePacker
        |
        | DeviceToDevice copy or pack kernel
        v
cudaArray / CUDA 3D texture object
        |
        | CUDA ray marcher samples tex3D<float>()
        v
OpenGL PBO mapped as CUDA device pointer
        |
        | unmap
        v
OpenGL texture update / fullscreen draw
```

重要原则：

```text
1. 不把 volume 每帧读回 CPU。
2. simulation buffer 和 render texture 解耦。
3. renderer 消费“渲染用 volume”，不直接依赖具体 rule。
4. OpenGL 只显示 image，不参与 simulation。
```

## Simulation MVP 数据流

Single-channel Lenia：

```text
A_t:    float32 [nx, ny, nz]
A_hat:  complex [nx, ny, nz/2+1]
K_hat:  complex [nx, ny, nz/2+1]
U:      float32 [nx, ny, nz]

每一步：
  cufft R2C(A_t) -> A_hat
  pointwise multiply A_hat * K_hat -> U_hat
  cufft C2R(U_hat) -> U
  normalize inverse FFT by 1/(nx*ny*nz)
  update kernel: A_t = clamp(A_t + dt * G(U), 0, 1)
```

建议从 real-to-complex cuFFT 开始，而不是 complex-to-complex。3D Lenia 状态是实数，R2C 节省内存和时间。

## 基础类型建议

```cpp
struct VolumeDesc {
    int nx = 128;
    int ny = 128;
    int nz = 128;
    int channels = 1;
    float voxel_size = 1.0f;
};

struct DeviceVolume {
    VolumeDesc desc;
    std::vector<float*> channel;   // channel-major device buffers
};

struct SimState {
    VolumeDesc desc;
    DeviceVolume A;                // current state
    DeviceVolume U;                // potentials / scratch
    // FFT buffers and plans owned by concrete simulation class
};

struct RenderMapping {
    int density_channel = 0;
    int emission_channel = -1;
    int color_channel = -1;
    float density_scale = 4.0f;
    float emission_scale = 1.0f;
};
```

第一版可以只实现一个 channel，但文件/类命名从一开始就预留多通道。

## 推荐目录结构

```text
VolLenia_Playground/
  CMakeLists.txt
  CMakePresets.json
  README.md
  docs/
  plans/
  configs/
    app.default.json
    lenia.default.json
  assets/
    transfer_functions/
    seeds/
  external/ or third_party/
  src/
    main.cpp
    app/
      App.h
      App.cpp
      Camera.h
      Camera.cpp
      UiPanel.h
      UiPanel.cpp
    core/
      VolumeDesc.h
      CudaCheck.h
      GlCheck.h
      MathTypes.h
    sim/
      Rule.h
      LeniaRule.h
      LeniaRule.cu
      KernelGenerator.h
      KernelGenerator.cu
      CufftContext.h
      CufftContext.cpp
      DeviceVolume.h
      DeviceVolume.cu
    render/
      CudaVolumeRenderer.h
      CudaVolumeRenderer.cu
      RenderParams.h
      RenderVolumePacker.h
      RenderVolumePacker.cu
      GlDisplay.h
      GlDisplay.cpp
      TransferFunction.h
      TransferFunction.cpp
    io/
      Config.h
      Config.cpp
      Lenia3DImportNotes.md
  tests/
  scripts/
```

## Rule 插件接口

不要让 App 直接写死 Lenia。建议从一开始用抽象接口：

```cpp
class Rule {
public:
    virtual ~Rule() = default;
    virtual void reset(const VolumeDesc& desc) = 0;
    virtual void step(float dt) = 0;
    virtual DeviceVolume& state() = 0;
    virtual const DeviceVolume& state() const = 0;
};
```

后续可以添加：

```text
LeniaRule
MultiChannelLeniaRule
SmoothLife3DRule
SpectralLogitLeniaRule
EcoLeniaRule
FlowLeniaLikeRule
NeuralGrowthRule
```

## Kernel 抽象

第一版可以只有一个 kernel，但建议参数结构面向扩展：

```cpp
struct KernelParams {
    float radius = 12.0f;
    int core_type = 1;               // polynomial / bump / step / gaussian-shell later
    std::vector<float> shell_weights;
    bool normalize_sum = true;
    bool zero_mean = false;
};
```

未来多通道不要立刻做 `C_out * C_in` 个 3D FFT kernel；建议使用 kernel basis：

```text
U[c,k] = K[k] * A[c]
V[o]   = sum_{c,k} W[o,c,k] U[c,k]
```

这样 kernel 个数可控，也适合后续 learnable kernel。

## Renderer 抽象

第一版 CUDA renderer：

```cpp
class CudaVolumeRenderer {
public:
    void resize(int width, int height);
    void setVolume(const DeviceVolume& state, const RenderMapping& mapping);
    void render(const Camera& camera, const RenderParams& params, GLuint pbo);
};
```

内部大致是：

```text
1. pack selected channel into cudaArray / texture object
2. map OpenGL PBO
3. launch raymarch kernel writing RGBA8 or uchar4
4. unmap PBO
```

第一版输出 `uchar4` 足够。后续 HDR/temporal accumulation 可以改成 float buffer。

## Render modes

先预留 enum：

```cpp
enum class VolumeRenderMode {
    EmissionAbsorption,
    MaxIntensityProjection,
    FirstHit,
    IsoSurfacePreview,
    DebugSlice
};
```

第一阶段只必须完成 `EmissionAbsorption`。

## 第一版 compositing 公式

对每个 ray sample：

```cpp
float density = sampleVolume(pos);
float sigma = density_scale * max(density - threshold, 0.0f);
float alpha = 1.0f - expf(-sigma * step_size);
float3 color = transferFunction(density);
accum.rgb += transmittance * alpha * color;
transmittance *= (1.0f - alpha);
if (transmittance < cutoff) break;
```

这样 step size 改变时比 `alpha = density * scale` 稳定。

## Windows/CUDA/OpenGL interop 注意点

```text
1. OpenGL context 必须先创建，再注册 PBO 给 CUDA。
2. 避免在 PBO 被 CUDA map 时让 OpenGL 访问它。
3. map/unmap 每帧一次，先求正确再优化。
4. 混合显卡环境下确保 OpenGL context 跑在 NVIDIA GPU。
5. resize window 时释放并重建 PBO/CUDA graphics resource。
6. 所有 CUDA API、kernel launch、OpenGL API 都要有 check 宏。
```

## 后期可扩展性原则

```text
1. Rule 可以换，renderer 不变。
2. renderer 可以换，Rule 不变。
3. UI 控制参数走 config/state，不直接写死 kernel 常量。
4. 每个 milestone 都必须有可运行 demo。
5. 不提前引入大型 renderer 框架。
```
