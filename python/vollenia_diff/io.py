from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
