"""Terrain interface (Phase 18).

The locomotion environment consumes `TerrainPatch` objects and never talks to a
specific generator. Any generator (procedural_rough today, Tomasz's Everest
generator later, e.g. `everest_south_col`) plugs in by producing a TerrainPatch
from a TerrainSpec. Replacing the generator must not require touching the
environment code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TerrainSpec:
    """Request for a terrain patch. Deterministic given (seed, params)."""

    seed: int = 0
    difficulty: float = 0.5          # 0..1, generator-defined meaning
    size_m: tuple[float, float] = (20.0, 20.0)
    resolution_m: float = 0.05       # grid cell edge length
    slope_deg: float = 0.0           # mean grade along +x
    roughness_m: float = 0.05        # height-noise amplitude
    obstacle_params: dict = field(default_factory=dict)
    friction_range: tuple[float, float] | None = None  # per-cell friction map if set
    metadata: dict = field(default_factory=dict)

    @property
    def grid_shape(self) -> tuple[int, int]:
        nx = int(round(self.size_m[0] / self.resolution_m)) + 1
        ny = int(round(self.size_m[1] / self.resolution_m)) + 1
        return nx, ny


@dataclass
class TerrainPatch:
    """A generated terrain: heightfield + optional friction map + metadata."""

    spec: TerrainSpec
    heights_m: np.ndarray            # (nx, ny) elevation grid
    friction: np.ndarray | None = None  # (nx, ny) per-cell sliding friction, or None
    metadata: dict = field(default_factory=dict)

    def height_at(self, x: float, y: float) -> float:
        """Bilinear height lookup; patch is centered at origin."""
        sx, sy = self.spec.size_m
        nx, ny = self.heights_m.shape
        fx = np.clip((x + sx / 2) / sx * (nx - 1), 0, nx - 1.000001)
        fy = np.clip((y + sy / 2) / sy * (ny - 1), 0, ny - 1.000001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        h = self.heights_m
        return float(
            h[i, j] * (1 - tx) * (1 - ty)
            + h[i + 1, j] * tx * (1 - ty)
            + h[i, j + 1] * (1 - tx) * ty
            + h[i + 1, j + 1] * tx * ty
        )


class TerrainGenerator(Protocol):
    def __call__(self, spec: TerrainSpec) -> TerrainPatch: ...


_REGISTRY: dict[str, TerrainGenerator] = {}


def register_generator(name: str, gen: TerrainGenerator) -> None:
    _REGISTRY[name] = gen


def get_generator(name: str) -> TerrainGenerator:
    if name not in _REGISTRY:
        raise KeyError(f"unknown terrain generator {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
