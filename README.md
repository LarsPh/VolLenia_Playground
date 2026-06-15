# VolLenia Playground

VolLenia Playground is a C++20/CUDA/OpenGL sandbox for future volumetric 3D Lenia experiments.

Milestone 0 only bootstraps the app shell:

- GLFW window titled `VolLenia Playground`
- Dear ImGui status panel
- CUDA runtime device query
- OpenGL clear-color background

It intentionally does not implement CUDA/OpenGL PBO interop, volume rendering, cuFFT, or Lenia simulation yet.

## Dependencies

The project uses CMake `FetchContent` instead of vendoring third-party source in the repository. Configure downloads these dependencies into the build tree:

- GLFW `3.4`
- Dear ImGui `v1.92.8`
- glad `v2.0.8`, generated as OpenGL 3.3 core
- nlohmann/json `v3.11.3`

CUDA is found through the local CUDA Toolkit with `find_package(CUDAToolkit REQUIRED)`.

GLAD's generator needs Python `jinja2`; CMake creates a build-local virtual environment under `build/python-venv` for that generator so the user's global Python environment is not modified.

## Environment

Recommended checks:

```powershell
nvidia-smi
nvcc --version
cmake --version
where cl
```

If `cl` is not visible in the current shell, load the Visual Studio x64 toolchain first. On this machine, the direct batch file path is:

```powershell
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && where cl"
```

## Build

Default Visual Studio generator build:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release
.\build\Release\VolLenia_Playground.exe
```

Equivalent preset workflow:

```powershell
cmake --preset vs2022-x64
cmake --build --preset release
.\build\Release\VolLenia_Playground.exe
```

If `native` CUDA architecture is not supported by the installed CMake/CUDA combination, specify the architecture manually. For an RTX 4060 Ti, use Ada Lovelace `sm_89`:

```powershell
cmake -S . -B build-sm89 -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-sm89 --config Release
.\build-sm89\Release\VolLenia_Playground.exe
```

## Configuration

The app reads `configs/app.default.json` when present. If the file is missing, built-in defaults are used. Invalid JSON is treated as a startup error.

## Lenia3D reference cells

The repository commits the Lenia3D catalog manifest at `configs/lenia3d_reference/animals.json`, but raw `.f32` cell assets under `assets/cells/lenia3d_reference/` are generated data and are ignored by Git.

After a fresh clone, regenerate the `.f32` files from the committed manifest:

```powershell
uv run python scripts/convert_lenia3d_animals.py --manifest configs\lenia3d_reference\animals.json --cells-dir assets\cells\lenia3d_reference
```

This script currently uses only the Python standard library. `uv run python` is the recommended launcher for consistency, but the project does not need a `pyproject.toml` yet. If the Python tooling grows real third-party dependencies later, adding `pyproject.toml` would be the right time to make that environment explicit.

To refresh the manifest from the external Lenia3D reference checkout, pass `--input`:

```powershell
uv run python scripts\convert_lenia3d_animals.py --input D:\projects\Lenia3D\src\data\animals3D.js --manifest configs\lenia3d_reference\animals.json --cells-dir assets\cells\lenia3d_reference --limit 0
```

## References

Local PDFs are kept under `references/` for offline reading:

- Bert Wang-Chak Chan, **Lenia - Biology of Artificial Life**: [arXiv:1812.05433](https://arxiv.org/abs/1812.05433) (`references/lenia.pdf`)
- Bert Wang-Chak Chan, **Lenia and Expanded Universe**: [arXiv:2005.03742](https://arxiv.org/abs/2005.03742) (`references/lenia_expanded.pdf`)
- Stephan Rafler, **Generalization of Conway's "Game of Life" to a continuous domain - SmoothLife**: [arXiv:1111.1567](https://arxiv.org/abs/1111.1567) (`references/smoothlife.pdf`)
