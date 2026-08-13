"""Built-in procedural terrain generators (deterministic per seed)."""

from __future__ import annotations

import numpy as np

from everest_locomotion.terrains.spec import TerrainPatch, TerrainSpec, register_generator


def flat(spec: TerrainSpec) -> TerrainPatch:
    nx, ny = spec.grid_shape
    return TerrainPatch(spec=spec, heights_m=np.zeros((nx, ny)))


def procedural_rough(spec: TerrainSpec) -> TerrainPatch:
    """Sloped plane + multi-octave value noise + optional discrete obstacles.

    difficulty scales roughness and obstacle height linearly.
    """
    rng = np.random.default_rng(spec.seed)
    nx, ny = spec.grid_shape
    xs = np.linspace(-spec.size_m[0] / 2, spec.size_m[0] / 2, nx)

    # base slope along +x
    heights = np.tan(np.deg2rad(spec.slope_deg)) * xs[:, None] * np.ones((nx, ny))

    # multi-octave noise: coarse undulation + fine roughness
    amp = spec.roughness_m * spec.difficulty
    for cells, w in ((16, 0.6), (8, 0.3), (4, 0.1)):
        coarse = rng.uniform(-1, 1, (max(nx // cells, 2), max(ny // cells, 2)))
        # bilinear upsample to grid
        ci = np.linspace(0, coarse.shape[0] - 1, nx)
        cj = np.linspace(0, coarse.shape[1] - 1, ny)
        i0 = np.clip(ci.astype(int), 0, coarse.shape[0] - 2)
        j0 = np.clip(cj.astype(int), 0, coarse.shape[1] - 2)
        ti = (ci - i0)[:, None]
        tj = (cj - j0)[None, :]
        up = (
            coarse[i0][:, j0] * (1 - ti) * (1 - tj)
            + coarse[i0 + 1][:, j0] * ti * (1 - tj)
            + coarse[i0][:, j0 + 1] * (1 - ti) * tj
            + coarse[i0 + 1][:, j0 + 1] * ti * tj
        )
        heights += amp * w * up

    # discrete box obstacles
    obs = spec.obstacle_params
    n_boxes = int(obs.get("n_boxes", 0))
    box_h = float(obs.get("box_height_m", 0.1)) * spec.difficulty
    box_sz = float(obs.get("box_size_m", 0.5))
    half = int(box_sz / spec.resolution_m / 2)
    for _ in range(n_boxes):
        ci = rng.integers(half, nx - half)
        cj = rng.integers(half, ny - half)
        heights[ci - half : ci + half, cj - half : cj + half] += rng.uniform(0.3, 1.0) * box_h

    friction = None
    if spec.friction_range is not None:
        lo, hi = spec.friction_range
        # smooth per-region friction map (proxy for ice/snow/rock patches)
        coarse = rng.uniform(lo, hi, (max(nx // 20, 2), max(ny // 20, 2)))
        fi = np.clip(np.linspace(0, coarse.shape[0] - 1, nx).astype(int), 0, coarse.shape[0] - 1)
        fj = np.clip(np.linspace(0, coarse.shape[1] - 1, ny).astype(int), 0, coarse.shape[1] - 1)
        friction = coarse[fi][:, fj]

    return TerrainPatch(spec=spec, heights_m=heights, friction=friction)


register_generator("flat", flat)
register_generator("procedural_rough", procedural_rough)
