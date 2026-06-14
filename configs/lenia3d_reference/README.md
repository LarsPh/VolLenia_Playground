# Lenia3D Reference Catalog

This directory contains the generated Lenia3D `animals3D.js` organism catalog entries that have both `params` and `cells`.

Regenerate the full imported catalog with:

```powershell
uv run python scripts/convert_lenia3d_animals.py --input D:\projects\Lenia3D\src\data\animals3D.js --manifest configs\lenia3d_reference\animals.json --cells-dir assets\cells\lenia3d_reference --limit 0
```

The `.f32` files use little-endian float32 values in the project layout:

```text
index = (z * ny + y) * nx + x
```

Imported cell volumes are center-padded into the simulation grid at runtime; they are not stretched.
