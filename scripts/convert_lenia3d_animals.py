#!/usr/bin/env python3
"""Convert Lenia3D animals3D.js entries into a JSON manifest and raw f32 cells."""

from __future__ import annotations

import argparse
import json
import re
import struct
from fractions import Fraction
from pathlib import Path


DIM_DELIM = {0: "", 1: "$", 2: "%", 3: "#", 4: "@A", 5: "@B", 6: "@C", 7: "@D", 8: "@E", 9: "@F"}
DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(r"D:\projects\Lenia3D\src\data\animals3D.js"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/lenia3d_reference/animals.json"))
    parser.add_argument("--cells-dir", type=Path, default=Path("assets/cells/lenia3d_reference"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of animals to export; 0 exports all.")
    return parser.parse_args()


def extract_animal_array(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"export\s+const\s+animalArr\s*=\s*(\[.*\])\s*;?\s*$", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find animalArr in {path}")
    return json.loads(match.group(1))


def append_stack(target: list, value, count_text: str, is_repeat: bool) -> None:
    target.append(value)
    if count_text:
        repeated = value if is_repeat else []
        for _ in range(int(count_text) - 1):
            target.append(repeated)


def ch_to_value(text: str) -> float:
    if text in {".", "b", ""}:
        return 0.0
    if text == "o":
        return 1.0
    if len(text) == 1:
        return float(ord(text) - ord("A") + 1) / 255.0
    return float((ord(text[0]) - ord("p")) * 24 + (ord(text[1]) - ord("A") + 25)) / 255.0


def max_nested_lengths(level: int, values, max_lens: list[int]) -> None:
    if level >= DIM:
        return
    max_lens[level] = max(max_lens[level], len(values))
    for item in values:
        if isinstance(item, list):
            max_nested_lengths(level + 1, item, max_lens)


def cubify(level: int, values, max_lens: list[int]) -> None:
    if level >= DIM:
        return
    fill = 0.0 if level == DIM - 1 else []
    while len(values) < max_lens[level]:
        values.append(fill)
    for item in values:
        if isinstance(item, list):
            cubify(level + 1, item, max_lens)


def rle_to_array(text: str):
    stacks = [[] for _ in range(DIM)]
    last = ""
    count = ""
    delims = list(DIM_DELIM.values())
    source = text.rstrip("!") + DIM_DELIM[DIM - 1]
    for ch in source:
        if ch.isdigit():
            count += ch
        elif ch in "pqrstuvwxy@":
            last = ch
        else:
            token = last + ch
            if token not in delims:
                append_stack(stacks[0], ch_to_value(token), count, True)
            else:
                dim = delims.index(token)
                for d in range(dim):
                    append_stack(stacks[d + 1], stacks[d], count, False)
                    stacks[d] = []
            last = ""
            count = ""

    cells = stacks[DIM - 1]
    max_lens = [0 for _ in range(DIM)]
    max_nested_lengths(0, cells, max_lens)
    cubify(0, cells, max_lens)
    return cells, [max_lens[2], max_lens[1], max_lens[0]]


def parse_shell_weights(text: str) -> list[float]:
    weights: list[float] = []
    for part in text.split(","):
        weights.append(float(Fraction(part.strip())))
    return weights


def slugify(name: str, used: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        slug = "animal"
    original = slug
    suffix = 2
    while slug in used:
        slug = f"{original}_{suffix}"
        suffix += 1
    used.add(slug)
    return slug


def flatten_x_fastest(cells) -> list[float]:
    flat: list[float] = []
    for z_slice in cells:
        for y_row in z_slice:
            for value in y_row:
                flat.append(float(value))
    return flat


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack(f"<{len(values)}f", *values))


def convert(args: argparse.Namespace) -> dict:
    source_entries = extract_animal_array(args.input)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.cells_dir.mkdir(parents=True, exist_ok=True)

    animals: list[dict] = []
    used_slugs: set[str] = set()
    for source_index, entry in enumerate(source_entries):
        if not isinstance(entry, dict) or "params" not in entry or "cells" not in entry:
            continue
        if args.limit > 0 and len(animals) >= args.limit:
            break

        cells, dims = rle_to_array(entry["cells"])
        flat = flatten_x_fastest(cells)
        expected = dims[0] * dims[1] * dims[2]
        if len(flat) != expected:
            raise ValueError(f"{entry.get('name', source_index)} decoded to {len(flat)} values, expected {expected}")

        display_name = str(entry.get("name", "")).strip()
        if not display_name:
            display_name = f"Animal #{len(animals)}"
        slug_source = display_name if display_name else str(entry.get("code", f"animal_{source_index}"))
        slug = slugify(slug_source, used_slugs)
        cells_path = args.cells_dir / f"{slug}.f32"
        write_f32(cells_path, flat)
        rel_cells_path = cells_path.relative_to(args.manifest.parent).as_posix() if cells_path.is_relative_to(args.manifest.parent) else cells_path.as_posix()
        try:
            rel_cells_path = cells_path.resolve().relative_to(args.manifest.parent.resolve()).as_posix()
        except ValueError:
            rel_cells_path = Path("../../") / cells_path
            rel_cells_path = rel_cells_path.as_posix()

        params = entry["params"]
        animal = {
            "id": len(animals),
            "source_index": source_index,
            "slug": slug,
            "code": entry.get("code", ""),
            "name": display_name,
            "cname": entry.get("cname", ""),
            "dims": dims,
            "cells_file": rel_cells_path,
            "params": {
                "R": float(params.get("R", 10.0)),
                "T": float(params.get("T", 10.0)),
                "b": parse_shell_weights(str(params.get("b", "1"))),
                "m": float(params.get("m", params.get("mu", 0.15))),
                "s": float(params.get("s", params.get("sigma", 0.015))),
                "kn": int(params.get("kn", 1)),
                "gn": int(params.get("gn", 1)),
            },
        }
        animals.append(animal)

    manifest = {
        "source": str(args.input),
        "format_version": 1,
        "layout": "x-fastest",
        "animals": animals,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = convert(args)
    print(f"Wrote {len(manifest['animals'])} animals to {args.manifest}")


if __name__ == "__main__":
    main()
