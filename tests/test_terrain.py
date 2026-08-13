"""Terrain interface tests: determinism, shapes, MuJoCo scene injection."""

import numpy as np

from everest_locomotion.robots.manifest import load_manifest
from everest_locomotion.sim_adapters.mujoco_scene import model_with_terrain
from everest_locomotion.terrains import TerrainSpec, flat, procedural_rough
from everest_locomotion.terrains.spec import get_generator


def test_deterministic_per_seed():
    spec = TerrainSpec(seed=42, difficulty=0.7, roughness_m=0.1)
    a = procedural_rough(spec)
    b = procedural_rough(spec)
    np.testing.assert_array_equal(a.heights_m, b.heights_m)
    c = procedural_rough(TerrainSpec(seed=43, difficulty=0.7, roughness_m=0.1))
    assert not np.array_equal(a.heights_m, c.heights_m)


def test_grid_shape_and_difficulty_scaling():
    spec = TerrainSpec(size_m=(10, 10), resolution_m=0.1)
    patch = flat(spec)
    assert patch.heights_m.shape == spec.grid_shape == (101, 101)
    rough_easy = procedural_rough(TerrainSpec(seed=1, difficulty=0.1))
    rough_hard = procedural_rough(TerrainSpec(seed=1, difficulty=1.0))
    assert rough_hard.heights_m.std() > rough_easy.heights_m.std()


def test_slope():
    patch = procedural_rough(TerrainSpec(seed=0, difficulty=0.0, slope_deg=10.0, size_m=(10, 10)))
    dx = patch.height_at(4.0, 0.0) - patch.height_at(-4.0, 0.0)
    assert abs(dx - 8.0 * np.tan(np.deg2rad(10.0))) < 0.05


def test_friction_map():
    patch = procedural_rough(TerrainSpec(seed=0, friction_range=(0.2, 0.9)))
    assert patch.friction is not None
    assert 0.2 <= patch.friction.min() and patch.friction.max() <= 0.9


def test_registry():
    assert get_generator("procedural_rough") is procedural_rough


def test_mujoco_scene_injection():
    manifest = load_manifest("a3_ultra")
    patch = procedural_rough(TerrainSpec(seed=3, difficulty=0.5, size_m=(8, 8)))
    model = model_with_terrain(manifest, patch)
    import mujoco

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") < 0  # replaced
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)  # compiles + no errors
