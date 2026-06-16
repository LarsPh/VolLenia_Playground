#!/usr/bin/env python3
"""Plot Lenia3D reference kernel profiles and function templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by missing env
    raise SystemExit(
        "Missing Python plotting dependency. Run `uv sync`, then rerun with "
        "`uv run python scripts/plot_lenia_kernel_profiles.py ...`."
    ) from exc


DEFAULT_ANIMAL_DIR = Path("assets/visualizations/lenia3d_reference/animals")
DEFAULT_FUNCTION_DIR = Path("assets/visualizations/lenia3d_reference/function_profiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/lenia3d_reference/animals.json"))
    parser.add_argument("--animal-index", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Generate plots for all animals in the manifest.")
    parser.add_argument("--limit", type=int, default=0, help="Generate plots for the first N animals; 0 means no limit.")
    parser.add_argument("--size", type=int, default=128, help="Cubic kernel canvas size.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANIMAL_DIR)
    parser.add_argument("--function-output-dir", type=Path, default=DEFAULT_FUNCTION_DIR)
    parser.add_argument("--format", choices=("jpg", "jpeg"), default="jpg")
    parser.add_argument("--write-json", action="store_true", help="Write radial profile JSON next to animal plots.")
    parser.add_argument("--write-origin-slice", action="store_true", help="Also write the FFT-origin z=0 kernel slice.")
    parser.add_argument("--write-function-profiles", action="store_true")
    return parser.parse_args()


def kernel_core(r: np.ndarray, kn: int) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32)
    inside = (r >= 0.0) & (r <= 1.0)
    out = np.zeros_like(r, dtype=np.float32)
    if kn == 2:
        valid = inside & (r > 0.0) & (r < 1.0)
        rv = r[valid]
        out[valid] = np.exp(4.0 - 1.0 / (rv * (1.0 - rv)))
    elif kn == 3:
        out[(r >= 0.25) & (r <= 0.75)] = 1.0
    elif kn == 4:
        out[(r >= 0.25) & (r <= 0.75)] = 1.0
        out[(r >= 0.0) & (r < 0.25)] = 0.5
    else:
        v = 4.0 * r[inside] * (1.0 - r[inside])
        out[inside] = v**4
    return out


def growth_function(u: np.ndarray, gn: int, mu: float, sigma: float) -> np.ndarray:
    sigma_safe = max(float(sigma), 1.0e-5)
    diff = u - float(mu)
    if gn == 2:
        return 2.0 * np.exp(-(diff * diff) / (2.0 * sigma_safe * sigma_safe)) - 1.0
    if gn == 3:
        return np.where(np.abs(diff) <= sigma_safe, 1.0, -1.0)
    value = np.maximum(0.0, 1.0 - (diff * diff) / (9.0 * sigma_safe * sigma_safe))
    return 2.0 * value**4 - 1.0


def build_kernel(size: int, params: dict[str, Any]) -> np.ndarray:
    radius = max(float(params.get("R", 10.0)), 1.0e-5)
    shell_weights = [float(v) for v in params.get("b", [1.0])]
    shell_count = max(1, len(shell_weights))
    kn = int(params.get("kn", 1))

    axis = np.arange(size, dtype=np.float32)
    wrapped = np.minimum(axis, size - axis)
    z, y, x = np.meshgrid(wrapped, wrapped, wrapped, indexing="ij")
    distance = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    q = distance / radius

    kernel = np.zeros((size, size, size), dtype=np.float32)
    mask = q < 1.0
    shell_position = q[mask] * float(shell_count)
    shell_index = np.clip(np.floor(shell_position).astype(np.int32), 0, shell_count - 1)
    local_r = shell_position - np.floor(shell_position)
    weights = np.asarray(shell_weights, dtype=np.float32)
    kernel[mask] = weights[shell_index] * kernel_core(local_r, kn)

    total = float(kernel.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("Kernel sum is zero; cannot normalize")
    kernel /= total
    return kernel


def radial_profile(size: int, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    radius = max(float(params.get("R", 10.0)), 1.0e-5)
    shell_weights = [float(v) for v in params.get("b", [1.0])]
    shell_count = max(1, len(shell_weights))
    kn = int(params.get("kn", 1))
    q = np.linspace(0.0, 1.0, max(size, 128), dtype=np.float32)
    shell_position = q * float(shell_count)
    shell_index = np.clip(np.floor(shell_position).astype(np.int32), 0, shell_count - 1)
    local_r = shell_position - np.floor(shell_position)
    local_r[-1] = 1.0
    weights = np.asarray(shell_weights, dtype=np.float32)
    values = weights[shell_index] * kernel_core(local_r, kn)
    return q * radius, values


def save_figure(path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, format=fmt, dpi=160, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close()


def animal_label(animal: dict[str, Any]) -> str:
    name = animal.get("name") or f"Animal #{animal.get('id', '?')}"
    code = animal.get("code") or "no-code"
    return f"{name} [{code}]"


def plot_animal(
    animal: dict[str, Any],
    size: int,
    output_dir: Path,
    fmt: str,
    write_json: bool,
    write_origin_slice: bool,
) -> list[Path]:
    slug = animal.get("slug") or f"animal_{animal.get('id', 0)}"
    params = animal.get("params", {})
    kernel = build_kernel(size, params)
    distances, values = radial_profile(size, params)
    title = animal_label(animal)
    suffix = "jpg" if fmt == "jpeg" else fmt
    paths: list[Path] = []

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(distances, values, linewidth=2.0)
    plt.xlabel("wrapped distance")
    plt.ylabel("unnormalized shell core value")
    plt.title(f"{title} radial kernel profile")
    plt.grid(alpha=0.25)
    path = output_dir / f"{slug}_radial_profile.{suffix}"
    save_figure(path, fmt)
    paths.append(path)

    vmax = max(float(kernel.max()), 1.0e-12)
    if write_origin_slice:
        origin_slice = kernel[0, :, :]
        plt.figure(figsize=(5.2, 4.8))
        plt.imshow(origin_slice, cmap="magma", vmin=0.0, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"{title} kernel origin slice z=0")
        plt.axis("off")
        path = output_dir / f"{slug}_kernel_origin_slice.{suffix}"
        save_figure(path, fmt)
        paths.append(path)

    centered = np.fft.fftshift(kernel)
    plt.figure(figsize=(5.2, 4.8))
    plt.imshow(centered[size // 2, :, :], cmap="magma", vmin=0.0, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title} kernel centered slice")
    plt.axis("off")
    path = output_dir / f"{slug}_kernel_centered_slice.{suffix}"
    save_figure(path, fmt)
    paths.append(path)

    if write_json:
        json_path = output_dir / f"{slug}_kernel_profile.json"
        json_path.write_text(
            json.dumps(
                {
                    "animal_id": animal.get("id"),
                    "slug": slug,
                    "distance": distances.tolist(),
                    "value": values.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths.append(json_path)
    return paths


def plot_function_profiles(output_dir: Path, fmt: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "jpg" if fmt == "jpeg" else fmt
    paths: list[Path] = []
    r = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    kernel_specs = [
        (1, "polynomial_bump"),
        (2, "exponential_bump"),
        (3, "step"),
        (4, "staircase"),
    ]
    for kn, name in kernel_specs:
        plt.figure(figsize=(6.0, 3.8))
        plt.plot(r, kernel_core(r, kn), linewidth=2.0)
        plt.ylim(-0.05, 1.1)
        plt.xlabel("local shell radius r")
        plt.ylabel("core(r)")
        plt.title(f"Kernel core kn={kn}: {name}")
        plt.grid(alpha=0.25)
        path = output_dir / f"kernel_core_kn{kn}_{name}.{suffix}"
        save_figure(path, fmt)
        paths.append(path)

    u = np.linspace(0.0, 0.35, 512, dtype=np.float32)
    mu = 0.15
    sigma = 0.015
    growth_specs = [
        (1, "polynomial"),
        (2, "gaussian"),
        (3, "step"),
    ]
    for gn, name in growth_specs:
        plt.figure(figsize=(6.0, 3.8))
        plt.plot(u, growth_function(u, gn, mu, sigma), linewidth=2.0)
        plt.axvline(mu, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
        plt.ylim(-1.1, 1.1)
        plt.xlabel("potential u")
        plt.ylabel("growth")
        plt.title(f"Growth gn={gn}: {name} (m={mu}, s={sigma})")
        plt.grid(alpha=0.25)
        path = output_dir / f"growth_gn{gn}_{name}.{suffix}"
        save_figure(path, fmt)
        paths.append(path)
    return paths


def selected_animals(animals: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all:
        return animals
    if args.limit > 0:
        return animals[: args.limit]
    if args.animal_index < 0 or args.animal_index >= len(animals):
        raise IndexError(f"--animal-index {args.animal_index} is out of range for {len(animals)} animals")
    return [animals[args.animal_index]]


def main() -> None:
    args = parse_args()
    root = json.loads(args.manifest.read_text(encoding="utf-8"))
    animals = root.get("animals", [])
    if not isinstance(animals, list) or not animals:
        raise ValueError(f"No animals found in {args.manifest}")

    written: list[Path] = []
    for animal in selected_animals(animals, args):
        written.extend(
            plot_animal(
                animal,
                args.size,
                args.output_dir,
                args.format,
                args.write_json,
                args.write_origin_slice,
            )
        )
    if args.write_function_profiles:
        written.extend(plot_function_profiles(args.function_output_dir, args.format))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
