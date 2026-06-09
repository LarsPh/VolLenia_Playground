# PLAN 01 — CUDA/OpenGL PBO smoke test

## 目标

验证 CUDA kernel 可以写入 OpenGL PBO，并通过 OpenGL fullscreen draw 显示。

这一阶段不渲染 volume，只渲染一个 CUDA 生成的动态图案。

## 目标文件

建议创建/修改：

```text
src/render/GlDisplay.h
src/render/GlDisplay.cpp
src/render/CudaPbo.h
src/render/CudaPbo.cpp
src/render/PboSmokeTest.h
src/render/PboSmokeTest.cu
src/core/CudaCheck.h
src/core/GlCheck.h
src/app/App.cpp
CMakeLists.txt
```

## 功能要求

```text
1. 创建 OpenGL PBO，大小等于 framebuffer width * height * uchar4。
2. 注册 PBO 给 CUDA：cudaGraphicsGLRegisterBuffer。
3. 每帧 map PBO，拿到 device pointer。
4. launch CUDA kernel 写 uchar4 gradient / animated pattern。
5. unmap PBO。
6. OpenGL 把 PBO 上传/绑定为 2D texture，画 fullscreen triangle。
7. window resize 时重建 PBO 和 GL texture。
```

## Smoke test pattern

建议 CUDA kernel 输出：

```text
x/y gradient
随 time 变化的圆形/波纹
alpha 固定 255
```

这样能肉眼确认每帧更新来自 CUDA。

## 约束

```text
1. 不做 volume texture。
2. 不接 cuFFT。
3. 不接 Lenia。
4. 不使用 glDrawPixels/fixed-function pipeline。
5. 不把 PBO 内容读回 CPU。
```

## 错误处理

必须检查：

```text
OpenGL shader compile/link errors
CUDA graphics resource register/map/unmap errors
CUDA kernel launch errors
```

## UI

ImGui 增加：

```text
render target size
CUDA/GL interop status
animated time
checkbox: enable smoke pattern
```

## 验收标准

```text
1. 运行后看到 CUDA 生成的动态图案。
2. resize 后仍显示正确。
3. Debug/Release 都能跑。
4. 关闭窗口时无 CUDA resource leak / crash。
```

## 建议 commit

```powershell
git add .
git commit -m "interop: add cuda opengl pbo smoke test"
```
