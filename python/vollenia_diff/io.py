from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .params import LeniaParams


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_params_from_manifest(path: Path, animal_index: int = 0) -> LeniaParams:
    root = read_json(path)
    animals = root.get("animals", [])
    if not isinstance(animals, list) or not animals:
        raise ValueError(f"No animals found in {path}")
    item = animals[animal_index]
    return LeniaParams.from_dict(item.get("params", {}))


@dataclass(slots=True)
class CatalogAnimalSource:
    manifest_path: Path
    animal: dict[str, Any]
    state: torch.Tensor
    params: LeniaParams
    source_metadata: dict[str, Any]


def _shape_zyx_from_dims(dims: list[int]) -> tuple[int, int, int]:
    if len(dims) != 3:
        raise ValueError(f"Expected dims [nx, ny, nz], got {dims}")
    return int(dims[2]), int(dims[1]), int(dims[0])


def _dims_from_shape_zyx(shape: tuple[int, int, int]) -> list[int]:
    return [int(shape[2]), int(shape[1]), int(shape[0])]


def _cubic_size_from_dims(dims: list[int]) -> int | None:
    if len(dims) != 3:
        return None
    values = [int(v) for v in dims]
    if values[0] == values[1] == values[2]:
        return values[0]
    return None


def _select_animal(animals: list[dict[str, Any]], selector: str | None) -> tuple[int, dict[str, Any]]:
    if not animals:
        raise ValueError("Catalog has no animals")
    if selector is None or selector == "":
        return 0, animals[0]
    try:
        index = int(selector)
        if 0 <= index < len(animals):
            return index, animals[index]
    except ValueError:
        pass
    needle = selector.lower()
    for index, animal in enumerate(animals):
        haystack = " ".join(
            str(animal.get(key, ""))
            for key in ("slug", "code", "name", "cname")
        ).lower()
        if needle in haystack:
            return index, animal
    raise ValueError(f"No animal matching {selector!r}")


def read_f32_cells(path: Path, dims: list[int]) -> torch.Tensor:
    shape = _shape_zyx_from_dims(dims)
    array = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"Expected {expected} float32 values in {path}, found {array.size}")
    return torch.from_numpy(array.reshape(shape).copy())


def center_pad_or_crop(state: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    output = torch.zeros(target_shape, device=state.device, dtype=state.dtype)
    src_slices = []
    dst_slices = []
    for source_size, target_size in zip(state.shape, target_shape, strict=True):
        copy_size = min(int(source_size), int(target_size))
        src_start = max((int(source_size) - copy_size) // 2, 0)
        dst_start = max((int(target_size) - copy_size) // 2, 0)
        src_slices.append(slice(src_start, src_start + copy_size))
        dst_slices.append(slice(dst_start, dst_start + copy_size))
    output[tuple(dst_slices)] = state[tuple(src_slices)]
    return output


def resample_state(state: torch.Tensor, target_shape: tuple[int, int, int], mode: str = "trilinear") -> torch.Tensor:
    if tuple(int(v) for v in state.shape) == tuple(int(v) for v in target_shape):
        return state
    volume = state.reshape(1, 1, *state.shape)
    if mode == "nearest":
        resized = F.interpolate(volume, size=target_shape, mode=mode)
    else:
        resized = F.interpolate(volume, size=target_shape, mode="trilinear", align_corners=False)
    return resized.reshape(target_shape)


def load_catalog_animal_state(
    manifest_path: Path,
    selector: str | None,
    *,
    size: int | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CatalogAnimalSource:
    root = read_json(manifest_path)
    animals = root.get("animals", [])
    if not isinstance(animals, list):
        raise ValueError(f"Invalid animals array in {manifest_path}")
    animal_index, animal = _select_animal(animals, selector)
    dims = [int(v) for v in animal.get("dims", [])]
    sim_dims = [int(v) for v in animal.get("simulation_dims", dims)]
    policy = str(animal.get("resolution_policy", "native"))
    cells_path = (manifest_path.parent / str(animal.get("cells_file", ""))).resolve()
    cells = read_f32_cells(cells_path, dims).to(device=device, dtype=dtype)
    if policy == "cropped":
        cells = center_pad_or_crop(cells, _shape_zyx_from_dims(sim_dims))
    native_size = _cubic_size_from_dims(sim_dims)
    target_size = int(size) if size is not None else native_size
    rule_radius_scale = 1.0
    if size is not None:
        target_shape = (int(size), int(size), int(size))
        cells = resample_state(cells, target_shape)
        if native_size is not None and native_size > 0:
            rule_radius_scale = float(size) / float(native_size)
    params = LeniaParams.from_dict(animal.get("params", {}))
    if rule_radius_scale != 1.0:
        params = replace(params, R=float(params.R) * rule_radius_scale)
    source_metadata = {
        "catalog": str(manifest_path),
        "animal_index": animal_index,
        "slug": animal.get("slug", ""),
        "code": animal.get("code", ""),
        "name": animal.get("name", ""),
        "dims": dims,
        "simulation_dims": sim_dims,
        "native_simulation_dims": sim_dims,
        "target_dims": _dims_from_shape_zyx(tuple(int(v) for v in cells.shape)),
        "target_size": target_size,
        "resolution_policy": policy,
        "rule_radius_scale": rule_radius_scale,
    }
    return CatalogAnimalSource(
        manifest_path=manifest_path,
        animal=animal,
        state=cells,
        params=params,
        source_metadata=source_metadata,
    )
