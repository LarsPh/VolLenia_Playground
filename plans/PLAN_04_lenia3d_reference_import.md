# PLAN 04 — Lenia3D reference import

## 目标

把 `D:\projects\Lenia3D` 中有用的 preset / kernel 参数 / animals 数据转成当前项目的 JSON config 或 seed 文件。

这个计划可以等 `PLAN_03` 通过后再执行。

## 参考文件

```text
D:\projects\Lenia3D\src\data\animals3D.js
D:\projects\Lenia3D\src\core\KernelGen.js
D:\projects\Lenia3D\src\core\LeniaEngine.js
```

## 输出文件建议

```text
configs/lenia3d_reference/README.md
configs/lenia3d_reference/presets/*.json
assets/seeds/lenia3d_reference/*.json 或 *.bin
src/io/Lenia3DImporter.h
src/io/Lenia3DImporter.cpp
scripts/convert_lenia3d_animals.py
```

## 工作步骤

```text
1. 阅读 animals3D.js 的数据格式，记录 RLE/shape/params。
2. 写 Python 脚本把 JS 数据转成中间 JSON。
3. 先支持 1-3 个 hand-picked animals，不追求全部导入。
4. 确认参数命名映射：R, b, kn, gn, m, s, T。
5. 在 UI 中添加 preset list。
6. reset 时加载 preset 的 seed + params。
```

## 注意

Lenia3D 的模拟 backend 是 TF.js，数值细节可能和 cuFFT CUDA 版不同。导入 preset 后不保证行为完全一致。

先追求：

```text
大致能跑
参数含义一致
可以作为搜索起点
```

不要追求 bit-exact parity。

## 验收标准

```text
1. 至少 3 个 Lenia3D-style preset 可从 UI 选择。
2. 每个 preset 能 reset 并运行。
3. 当前参数可保存为 JSON。
4. 文档记录哪些参数已映射、哪些还未支持。
```

## 建议 commit

```powershell
git add .
git commit -m "io: import initial lenia3d reference presets"
```
