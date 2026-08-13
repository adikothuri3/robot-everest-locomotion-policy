"""Build MuJoCo scenes that combine the A3 model with a generated TerrainPatch.

Replaces the model's flat floor with a heightfield produced by any terrain
generator, keeping relative mesh paths working by writing the composed XML next
to the original MJCF.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from everest_locomotion.robots.manifest import RobotManifest
from everest_locomotion.terrains.spec import TerrainPatch

SCENE_SUFFIX = "_scene_hfield.xml"


def model_with_terrain(manifest: RobotManifest, patch: TerrainPatch) -> mujoco.MjModel:
    tree = ET.parse(manifest.mjcf_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") == "floor":
            worldbody.remove(geom)

    asset = root.find("asset")
    h = patch.heights_m
    h_min, h_max = float(h.min()), float(h.max())
    elevation = max(h_max - h_min, 1e-6)
    nx, ny = h.shape
    ET.SubElement(
        asset,
        "hfield",
        name="terrain",
        nrow=str(ny),
        ncol=str(nx),
        size=(
            f"{patch.spec.size_m[0] / 2} {patch.spec.size_m[1] / 2} {elevation} 0.5"
        ),
    )
    mean_friction = 1.0
    if patch.friction is not None:
        mean_friction = float(np.mean(patch.friction))
    ET.SubElement(
        worldbody,
        "geom",
        name="terrain",
        type="hfield",
        hfield="terrain",
        conaffinity="7",
        friction=f"{mean_friction} 0.005 0.0001",
        pos=f"0 0 {h_min}",
    )

    out = manifest.mjcf_path.with_name(manifest.mjcf_path.stem + SCENE_SUFFIX)
    tree.write(out)
    model = mujoco.MjModel.from_xml_path(str(out))
    out.unlink(missing_ok=True)

    # fill heightfield data (row-major, normalized 0..1)
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    norm = (h - h_min) / elevation
    # MuJoCo hfield data layout: row r = y index, col c = x index
    model.hfield_data[:] = norm.T.ravel()
    return model
