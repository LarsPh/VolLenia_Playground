# PLAN 04 — Lenia3D biology catalog + imported cell seeds

## Goal

Import useful Lenia3D reference organisms from:

```text
D:\projects\Lenia3D\src\data\animals3D.js
D:\projects\Lenia3D\src\utils\load.js
D:\projects\Lenia3D\src\core\KernelGen.js
D:\projects\Lenia3D\src\core\LeniaEngine.js
```

The important shift from PLAN 03:

```text
PLAN 03:
  procedural seed preset + parameter preset are independent

PLAN 04:
  add coupled animal presets = cells + params + kernel core + growth function
  but still allow cells-only / params-only / manual fine tuning
```

This milestone should make it easy to explore more organisms quickly without losing the ability to experiment by mixing seed and rule parameters.

---

## Background: what “cells” means here

In Lenia3D, an animal entry contains:

```text
params: R, T, b, m, s, kn, gn
cells:  RLE-encoded initial 3D scalar field
```

The `cells` field is not a learned feature vector and not a cell type system. It is the initial density/state volume:

```text
A(x, y, z) in [0, 1]
```

Think of it as the organism’s “embryo” or starting body. Many Lenia creatures only survive or move when the initial cells are paired with the matching rule parameters, so importing cells is much more useful than using random procedural seeds only.

PLAN 04 should support this coupling, while still keeping experimental decoupling available.

---

## Reference mapping

Lenia3D JS uses these names:

```text
R   -> kernel radius
T   -> time scale, update uses dt = 1 / T
b   -> shell weights, sometimes fractions like "1,3/4,7/12,11/12"
m   -> growth mean / mu
s   -> growth width / sigma
kn  -> kernel core ID, 1-based in JS params
gn  -> growth function ID, 1-based in JS params
cells -> RLE-encoded initial scalar field
```

Current C++ code uses:

```text
radius, T, mu, sigma, shell_weights, shell_count
```

PLAN 04 should add:

```text
KernelCoreType / kernel_core_id
GrowthFunctionType / growth_function_id
```

At minimum support:

```text
Kernel core 1: polynomial bump       (4r(1-r))^4
Growth 1:     polynomial/quartic     2 * max(0, 1 - (u-m)^2/(9s^2))^4 - 1
Growth 2:     Gaussian               2 * exp(-(u-m)^2/(2s^2)) - 1
Growth 3:     step                   optional / debug only
```

Keep the existing Gaussian growth as an option, but imported Lenia3D animals with `gn = 1` should use the polynomial/quartic growth function unless the user overrides it.

---

## Target files

Suggested additions/modifications:

```text
scripts/convert_lenia3d_animals.py
configs/lenia3d_reference/README.md
configs/lenia3d_reference/animals.json
assets/cells/lenia3d_reference/*.f32

src/io/LeniaAnimalCatalog.h
src/io/LeniaAnimalCatalog.cpp
src/io/CellVolumeFile.h
src/io/CellVolumeFile.cpp

src/sim/LeniaParams.h
src/sim/LeniaSimulation.h
src/sim/LeniaSimulation.cu
src/sim/LeniaSeeds.h
src/sim/LeniaSeeds.cu

src/app/App.h
src/app/App.cpp
src/app/UiPanel.h
src/app/UiPanel.cpp
CMakeLists.txt
```

If this is too large for one pass, implement it in this order:

```text
1. Add data model + converter script
2. Load catalog manifest in C++
3. Apply animal params only
4. Apply animal cells only
5. Add Load animal = cells + params
6. Add fine-grain UI editing
```

---

## Data conversion design

### Converter script

Create:

```text
scripts/convert_lenia3d_animals.py
```

It should:

```text
1. Read D:\projects\Lenia3D\src\data\animals3D.js by default.
2. Extract entries with both `params` and `cells`.
3. Decode the RLE `cells` string by reimplementing the logic from `src/utils/load.js`.
4. Parse fractional shell weights such as `3/4`.
5. Write a manifest:
     configs/lenia3d_reference/animals.json
6. Write one raw little-endian float32 file per animal:
     assets/cells/lenia3d_reference/<slug>.f32
```

The manifest should look like:

```json
{
  "source": "Lenia3D animals3D.js",
  "format_version": 1,
  "animals": [
    {
      "id": 0,
      "slug": "diguttome_saliens",
      "code": "4Gu2s",
      "name": "Diguttome saliens",
      "cname": "乙雫球(躍)",
      "dims": [14, 14, 14],
      "cells_file": "../../assets/cells/lenia3d_reference/diguttome_saliens.f32",
      "params": {
        "R": 10.0,
        "T": 10.0,
        "b": [1.0, 0.75, 0.5833333, 0.9166667],
        "m": 0.12,
        "s": 0.01,
        "kn": 1,
        "gn": 1
      }
    }
  ]
}
```

### Cell axis/order

Use the project’s current x-fastest layout:

```cpp
index = (z * ny + y) * nx + x;
```

The Lenia3D RLE delimiters correspond roughly to:

```text
characters -> x row values
$          -> next y row
%          -> next z slice
```

The converter should flatten decoded arrays into x-fastest order and store `dims = [nx, ny, nz]`.

### Padding/placement

When applying imported cells to a simulation volume larger than the source cells, center-pad them, matching Lenia3D’s `resize()` behavior conceptually:

```text
clear target volume to 0
place source cells centered in target volume
no interpolation/resampling in first version
```

This is important. Do not stretch a 14^3 animal to fill 128^3. The reference JS pads small animals into the larger grid; it does not scale them to fill the box.

---

## Runtime data model

Add a small catalog type:

```cpp
struct LeniaAnimalPreset {
    int id = -1;
    std::string slug;
    std::string code;
    std::string name;
    std::string cname;
    VolumeDesc cells_desc;
    std::filesystem::path cells_file;
    LeniaParams params;
};
```

Add:

```cpp
class LeniaAnimalCatalog {
public:
    void load(const std::filesystem::path& manifest_path);
    int count() const;
    const LeniaAnimalPreset& animal(int index) const;
};
```

Add a loader for raw `.f32` cell files:

```cpp
class CellVolumeFile {
public:
    static DeviceVolume loadToDevice(const std::filesystem::path& path, VolumeDesc desc);
};
```

Simpler alternative: load the `.f32` into a temporary host vector, copy it into a temporary `DeviceVolume`, then launch a placement kernel.

---

## Simulation integration

Add support for imported cell seeds without removing procedural seeds.

Suggested API:

```cpp
void LeniaSimulation::resetImportedCells(DeviceVolumeView source_cells);
```

Implementation:

```text
1. Clear current state volume.
2. Launch CUDA kernel to copy source cells into the center of the target volume.
3. Clamp values to [0, 1].
4. Reset generation to 0.
```

This keeps the renderer and simulation generic.

---

## UI design

The UI should make the coupling explicit but editable.

### Animal catalog section

Add:

```text
Animal preset combo
Load animal          -> apply cells + params + kn/gn, reset generation
Apply cells only     -> reset current state with animal cells, keep current params
Apply params only    -> update params + rebuild kernel, keep current state
Reload manifest      -> optional
```

Add dirty indicators:

```text
Current animal: Diguttome saliens
Cells: from animal / procedural / modified
Params: from animal / modified
```

### Fine-grain parameter UI

Keep existing controls, but extend them:

```text
R slider
T slider
m / mu slider
s / sigma slider
kernel core combo: polynomial / exponential bump / step / staircase if implemented
growth function combo: polynomial / Gaussian / step
shell_count slider or combo
shell weight sliders b0..b7
Rebuild kernel button
Reset params to selected animal
```

Changing `R`, `kernel core`, `shell_count`, or shell weights should mark the kernel dirty.

Changing `T`, `m`, `s`, or growth function should not require rebuilding the kernel unless implementation stores growth data in the kernel.

### Resolution UI

Since 128^3 is currently fast, expose more choices:

```text
64, 96, 128, 160, 192, 256
```

Treat 192/256 as experimental. Changing resolution rebuilds simulation and reapplies the current seed/cells.

---

## Acceptance criteria

```text
1. Converter generates a manifest and .f32 cell files from Lenia3D animals3D.js.
2. App loads the manifest at startup if present.
3. UI lists at least the first 3 imported animals.
4. Load animal applies coupled cells + params and runs.
5. Apply cells only works.
6. Apply params only works.
7. Fine-grain sliders allow changing R/T/m/s/growth/kernel/shell weights.
8. Procedural seed presets still work.
9. Synthetic source still works.
10. No full-volume CPU readback from GPU state.
```

---

## Non-goals

```text
Do not implement multi-channel Lenia yet.
Do not implement environment/nutrient fields yet.
Do not try to achieve bit-exact parity with TF.js/WebGL.
Do not implement all renderer TODOs in this milestone.
Do not implement search/optimization yet.
```

---

## Suggested manual test flow

```powershell
python scripts/convert_lenia3d_animals.py `
  --input D:\projects\Lenia3D\src\data\animals3D.js `
  --manifest configs\lenia3d_reference\animals.json `
  --cells-dir assets\cells\lenia3d_reference `
  --limit 12

cmake --build build --config Release
.\build\Release\VolLenia_Playground.exe
```

Then test:

```text
1. Load Diguttome saliens.
2. Pause, single-step, play.
3. Apply Diguttome tardus params only to current cells.
4. Apply Triguttome labens cells only to current params.
5. Change growth function between polynomial and Gaussian.
6. Adjust shell weights and rebuild kernel.
7. Switch back to synthetic source.
```

---

## Suggested commit

```powershell
git add .
git commit -m "io: add lenia3d biology catalog and cell seeds"
```
