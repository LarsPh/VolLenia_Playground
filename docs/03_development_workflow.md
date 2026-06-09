# 03 — 开发工作流、Git 与 Codex 使用方式

## 本地初始化建议

PowerShell：

```powershell
cd D:\projects\VolLenia_Playground
git init
git checkout -b main

# 建议记录本地参考路径，当前 shell 生效
$env:CUDA_SAMPLES_ROOT="D:\projects\cuda_samples_13.3"
$env:LENIA3D_ROOT="D:\projects\Lenia3D"

# 可选：永久记录
setx CUDA_SAMPLES_ROOT "D:\projects\cuda_samples_13.3"
setx LENIA3D_ROOT "D:\projects\Lenia3D"
```

第一批 commit 建议只放文档：

```powershell
git add docs plans codex_prompts templates README_START_HERE.md
git commit -m "docs: add project plan"
```

之后每个 milestone 单独 commit，避免一次改太大。

## 环境检查

手动确认：

```powershell
nvidia-smi
nvcc --version
cmake --version
git --version
where cl
```

如果 `where cl` 找不到，先从 “x64 Native Tools Command Prompt for VS” 启动 shell，或确保 Visual Studio Build Tools 环境变量正确。

## 建议先单独验证 CUDA samples

在开始重写前，先确认本机能编译/运行 CUDA samples 的 volume rendering 示例。不同 CUDA samples 目录结构可能不同，用文件名搜索：

```powershell
cd D:\projects\cuda_samples_13.3
Get-ChildItem -Recurse -Filter volumeRender.cpp
Get-ChildItem -Recurse -Filter volumeRender_kernel.cu
```

如果原 sample 能跑，说明：

```text
CUDA toolkit
OpenGL interop
NVIDIA driver
Visual Studio toolchain
```

大概率都正常。

## Codex 工作方式建议

每次只给 Codex 一个 milestone 或更小任务。推荐格式：

```text
Context:
  当前项目是什么，已有文件是什么。

Goal:
  这一轮只实现什么。

Constraints:
  不做什么，不引入什么大型依赖。

Target files:
  明确要创建/修改哪些文件。

Acceptance criteria:
  必须能 build/run，UI 能显示什么。

After implementation:
  让 Codex 总结改了什么，给出 build commands。
```

本包的 `codex_prompts/` 已经写好前几个 prompt。

## 每个里程碑的审阅 checklist

```text
1. 是否能从 clean build 目录 configure/build？
2. 是否有明确运行命令？
3. 是否没有 CPU readback 大 volume？
4. 是否有 CUDA/OpenGL error checks？
5. 是否没有把 renderer 写死为 single Lenia channel？
6. 是否没有引入 Falcor/OptiX/pbrt 等过早依赖？
7. 是否保持每一步可运行、可截图、可调参？
```

## 推荐 commit 粒度

```text
docs: add project plan
build: add cmake skeleton
app: add glfw imgui window
cuda: add device info smoke test
interop: add cuda opengl pbo smoke test
render: add synthetic volume ray marcher
sim: add single channel lenia state buffers
sim: add cufft convolution
sim: add lenia growth update
io: add json configs
```

## Debug/诊断建议

```text
CUDA_LAUNCH_BLOCKING=1      # 仅调试时使用，会明显变慢
Nsight Systems              # 看 frame pipeline 和 GPU/CPU overlap
Nsight Compute              # 后期看 kernel 性能
RenderDoc                   # OpenGL display 层问题
```

Windows PowerShell 临时设置：

```powershell
$env:CUDA_LAUNCH_BLOCKING="1"
```

## 性能目标不要太早设死

第一阶段推荐先用：

```text
128^3 volume
512x512 render target
1 sim step / frame 或手动 step
fixed ray step size
single-channel
```

跑通后再加：

```text
192^3 / 256^3
多 sim steps per frame
multi-channel
gradient shading
empty-space skipping
```
