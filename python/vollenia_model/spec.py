from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ChannelSpec:
    name: str = "body"
    role: str = "matter"


@dataclass
class SmoothKernelBasis:
    center: float = 0.5
    width: float = 0.1
    amplitude: float = 1.0


@dataclass
class GrowthSpec:
    family: str = "gaussian"
    mu: float = 0.12
    sigma: float = 0.015


@dataclass
class KernelSpec:
    name: str = "kernel"
    src: int = 0
    dst: int = 0
    weight: float = 1.0
    family: str = "smooth_gaussian_mixture"
    radius: float = 12.0
    envelope_sharpness: float = 10.0
    basis: list[SmoothKernelBasis] = field(default_factory=list)
    growth: GrowthSpec = field(default_factory=GrowthSpec)
    shell_weights: list[float] = field(default_factory=list)


@dataclass
class FlowParams:
    theta_A: float = 1.0
    alpha_power: float = 2.0
    flow_max: float = 1.0
    transport_sigma: float = 0.5
    reintegration_dd: int = 1
    border: str = "torus"
    gradient: str = "sobel3d"


@dataclass
class ModelSpec:
    format_version: int = 1
    model_type: str = "expanded_flow"
    name: str = "model"
    dims: tuple[int, int, int] = (64, 64, 64)  # nx, ny, nz
    channels: list[ChannelSpec] = field(default_factory=list)
    render_channel: int = 0
    update_mode: str = "expanded_additive"
    dt: float = 0.1
    hard_clip: bool = True
    flow: FlowParams = field(default_factory=FlowParams)
    kernels: list[KernelSpec] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def kernel_count(self) -> int:
        return len(self.kernels)

    @property
    def matter_indices(self) -> list[int]:
        return [i for i, channel in enumerate(self.channels) if channel.role == "matter"]


def _dims_from_simulation(sim: dict[str, Any]) -> tuple[int, int, int]:
    if "dims" in sim:
        dims = sim["dims"]
        if not isinstance(dims, list | tuple) or len(dims) != 3:
            raise ValueError("simulation.dims must be [nx, ny, nz]")
        return int(dims[0]), int(dims[1]), int(dims[2])
    resolution = int(sim.get("resolution", 64))
    return resolution, resolution, resolution


def model_spec_from_dict(root: dict[str, Any], source_path: Path | None = None) -> ModelSpec:
    spec = ModelSpec(source_path=source_path)
    spec.format_version = int(root.get("format_version", spec.format_version))
    spec.model_type = str(root.get("model_type", spec.model_type))
    spec.name = str(root.get("name", source_path.stem if source_path else spec.name))
    spec.dims = _dims_from_simulation(root.get("simulation", {}))

    state = root.get("state", {})
    spec.render_channel = int(state.get("render_channel", spec.render_channel))
    spec.channels = [
        ChannelSpec(name=str(item.get("name", f"channel_{i}")), role=str(item.get("role", "matter")))
        for i, item in enumerate(state.get("channels", []))
    ]

    update = root.get("update", {})
    spec.update_mode = str(update.get("mode", spec.update_mode))
    spec.dt = float(update.get("dt", spec.dt))
    spec.hard_clip = str(update.get("clip", "hard")) == "hard"
    flow = update.get("flow", {})
    spec.flow = FlowParams(
        theta_A=float(flow.get("theta_A", 1.0)),
        alpha_power=float(flow.get("alpha_power", 2.0)),
        flow_max=float(flow.get("flow_max", 1.0)),
        transport_sigma=float(flow.get("transport_sigma", 0.5)),
        reintegration_dd=int(flow.get("reintegration_dd", 1)),
        border=str(flow.get("border", "torus")),
        gradient=str(flow.get("gradient", "sobel3d")),
    )

    kernels: list[KernelSpec] = []
    for i, item in enumerate(root.get("kernels", [])):
        basis = [
            SmoothKernelBasis(
                center=float(entry.get("center", 0.5)),
                width=float(entry.get("width", 0.1)),
                amplitude=float(entry.get("amplitude", 1.0)),
            )
            for entry in item.get("basis", [])
        ]
        growth = item.get("growth", {})
        kernels.append(
            KernelSpec(
                name=str(item.get("name", f"kernel_{i}")),
                src=int(item.get("src", 0)),
                dst=int(item.get("dst", 0)),
                weight=float(item.get("weight", 1.0)),
                family=str(item.get("family", "smooth_gaussian_mixture")),
                radius=float(item.get("R", item.get("radius", 12.0))),
                envelope_sharpness=float(item.get("envelope_sharpness", 10.0)),
                basis=basis,
                growth=GrowthSpec(
                    family=str(growth.get("family", "gaussian")),
                    mu=float(growth.get("mu", 0.12)),
                    sigma=float(growth.get("sigma", 0.015)),
                ),
                shell_weights=[float(v) for v in item.get("shell_weights", [])],
            )
        )
    spec.kernels = kernels
    validate_model_spec(spec)
    return spec


def validate_model_spec(spec: ModelSpec) -> None:
    if spec.format_version != 1:
        raise ValueError(f"Unsupported ModelSpec format_version: {spec.format_version}")
    if spec.model_type != "expanded_flow":
        raise ValueError(f"Unsupported ModelSpec model_type: {spec.model_type}")
    if any(d <= 0 for d in spec.dims):
        raise ValueError("ModelSpec dimensions must be positive")
    if not spec.channels:
        raise ValueError("ModelSpec must contain at least one channel")
    if not (0 <= spec.render_channel < spec.channel_count):
        raise ValueError("ModelSpec render_channel is out of range")
    if not spec.kernels:
        raise ValueError("ModelSpec must contain at least one kernel")
    if spec.update_mode not in {"expanded_additive", "flow"}:
        raise ValueError(f"Unsupported update mode: {spec.update_mode}")
    if spec.flow.transport_sigma != 0.5:
        raise ValueError("Stage 2 supports transport_sigma=0.5 only")
    for kernel in spec.kernels:
        if kernel.family not in {"smooth_gaussian_mixture", "legacy_shell"}:
            raise ValueError(f"Unsupported kernel family: {kernel.family}")
        if kernel.growth.family not in {"gaussian", "polynomial_lenia3d"}:
            raise ValueError(f"Unsupported growth family: {kernel.growth.family}")
        if not (0 <= kernel.src < spec.channel_count and 0 <= kernel.dst < spec.channel_count):
            raise ValueError(f"Kernel channel index out of range: {kernel.name}")
        if kernel.radius <= 0.0:
            raise ValueError(f"Kernel radius must be positive: {kernel.name}")
        if kernel.growth.sigma <= 0.0:
            raise ValueError(f"Growth sigma must be positive: {kernel.name}")


def load_model_spec(path: str | Path) -> ModelSpec:
    spec_path = Path(path)
    return model_spec_from_dict(json.loads(spec_path.read_text(encoding="utf-8")), spec_path)
