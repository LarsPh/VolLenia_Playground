# VolLenia Playground 启动文档包

这个文档包用于启动 `D:\projects\VolLenia_Playground` 项目。建议先把整个文档包解压到目标项目根目录，作为第一批 commit 的内容，然后按 `plans/` 里的任务卡逐步交给 Codex 实现。

## 项目一句话目标

做一个面向探索的 **Volumetric 3D Lenia Playground**：

```text
CUDA/cuFFT 负责 3D Lenia / 后续复杂模拟
CUDA ray marcher 负责实时体渲染
OpenGL/GLFW/ImGui 负责窗口、显示、交互 UI
Lenia3D repo 作为算法/参数/pattern 参考
cuda_samples volumeRender 作为 CUDA 体渲染/interop 参考
```

第一阶段不是做最完整的物理体渲染，也不是一开始接入 Falcor / OptiX / pbrt，而是先把下面这条最短 GPU 数据链跑通：

```text
3D volume state on GPU
  -> CUDA ray marching
  -> OpenGL PBO / fullscreen display
  -> ImGui 参数控制
```

然后再接：

```text
cuFFT Lenia simulation
  -> renderer consumes selected volume channel
  -> kernel/growth playground
  -> multi-channel / environment / spectral-logit / neural variants
```

## 推荐阅读顺序

1. `docs/00_project_brief.md`：项目背景、目标、非目标。
2. `docs/01_architecture.md`：第一阶段架构、GPU 数据流、主要接口。
3. `docs/02_roadmap.md`：阶段里程碑与验收标准。
4. `docs/03_development_workflow.md`：本地初始化、Git、Codex 工作流。
5. `plans/PLAN_00_bootstrap_repo.md`：第一张可执行任务卡。
6. `codex_prompts/CODEX_00_bootstrap_repo.md`：可直接复制给 Codex 的第一条指令。

## 已知本地参考路径

```text
D:\projects\cuda_samples_13.3      # 为了参考 volume renderer / CUDA-OpenGL interop
D:\projects\Lenia3D                # 为了参考 3D Lenia rule / kernel / patterns
D:\projects\VolLenia_Playground    # 目标项目
```

如果 CUDA samples 的目录结构和文档中写的不完全一致，优先用文件名搜索：

```text
volumeRender.cpp
volumeRender_kernel.cu
volumeRender.h
```

如果 Lenia3D 的目录结构变化，优先搜索：

```text
LeniaEngine.js
KernelGen.js
FftUtils.js
Renderer3D.js
animals3D.js
```

## 第一批建议 commit

```powershell
cd D:\projects\VolLenia_Playground
git init
git checkout -b main
# 把这个文档包里的 docs/, plans/, codex_prompts/, templates/ 解压/复制进来
git add .
git commit -m "docs: add VolLenia project plan"
```

之后开始执行 `PLAN_00`。
