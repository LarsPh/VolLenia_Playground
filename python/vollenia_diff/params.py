from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LeniaParams:
    """Single-channel Lenia parameters shared with the C++ catalog schema."""

    R: float = 10.0
    T: float = 10.0
    m: float = 0.12
    s: float = 0.01
    b: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5833333, 0.9166667])
    kn: int = 1
    gn: int = 1

    @property
    def mu(self) -> float:
        return self.m

    @mu.setter
    def mu(self, value: float) -> None:
        self.m = float(value)

    @property
    def sigma(self) -> float:
        return self.s

    @sigma.setter
    def sigma(self, value: float) -> None:
        self.s = float(value)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LeniaParams":
        if data is None:
            return cls()
        return cls(
            R=float(data.get("R", 10.0)),
            T=float(data.get("T", 10.0)),
            m=float(data.get("m", data.get("mu", 0.12))),
            s=float(data.get("s", data.get("sigma", 0.01))),
            b=[float(v) for v in data.get("b", [1.0, 0.75, 0.5833333, 0.9166667])],
            kn=int(data.get("kn", 1)),
            gn=int(data.get("gn", 1)),
        )

    def sanitized(self) -> "LeniaParams":
        weights = [float(v) for v in self.b]
        if not weights or all(v == 0.0 for v in weights):
            weights = [1.0]
        return LeniaParams(
            R=max(float(self.R), 1.0e-5),
            T=max(float(self.T), 1.0),
            m=float(self.m),
            s=max(float(self.s), 1.0e-5),
            b=weights,
            kn=int(self.kn),
            gn=int(self.gn),
        )

    def to_catalog_dict(self) -> dict[str, Any]:
        params = self.sanitized()
        return {
            "R": params.R,
            "T": params.T,
            "b": params.b,
            "m": params.m,
            "s": params.s,
            "kn": params.kn,
            "gn": params.gn,
        }
