"""The two generated assets and the Holosoma preset must describe one robot.

IsaacSim (the default training backend) imports the URDF and merges its fixed
joints; MuJoCo/MJWarp reads the MJCF; Holosoma asserts that the simulator's body
list equals `RobotConfig.body_names` exactly. A drift between any two of those
three is a startup crash on the cloud, so it is checked here instead.
"""

import ast
import xml.etree.ElementTree as ET

import mujoco
import pytest

from everest_locomotion import REPO_ROOT

ASSET_DIR = REPO_ROOT / "assets" / "a3_ultra" / "holosoma"
XML = ASSET_DIR / "a3_ultra_29dof.xml"
URDF = ASSET_DIR / "a3_ultra_29dof.urdf"

pytestmark = pytest.mark.skipif(
    not (XML.exists() and URDF.exists()), reason="generated asset missing"
)


def _preset_literal(name: str):
    src = (REPO_ROOT / "src/everest_locomotion/holosoma_ext/a3_ultra_presets.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in presets")


def _preset_robot_field(field: str):
    """Read a keyword value out of the `a3_ultra_29dof = RobotConfig(...)` call."""
    src = (REPO_ROOT / "src/everest_locomotion/holosoma_ext/a3_ultra_presets.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "a3_ultra_29dof" and isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == field:
                        return ast.literal_eval(kw.value)
    raise AssertionError(f"RobotConfig field {field} not found")


def _mjcf_bodies() -> list[str]:
    m = mujoco.MjModel.from_xml_path(str(XML))
    names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
    ]
    return [n for n in names if n != "world"]


def _urdf_bodies_after_merge() -> list[str]:
    """Links surviving a fixed-joint merge (IsaacLab/IsaacGym `dont_collapse`)."""
    root = ET.parse(URDF).getroot()
    merged = {
        j.find("child").get("link")
        for j in root.iter("joint")
        if j.get("type") == "fixed" and j.get("dont_collapse") != "true"
    }
    return [link.get("name") for link in root.iter("link") if link.get("name") not in merged]


def test_mjcf_body_order_matches_preset():
    assert _mjcf_bodies() == _preset_literal("BODY_NAMES")


def test_urdf_post_merge_bodies_match_preset():
    """Without this the IsaacSim backend aborts on its body_names assertion."""
    assert set(_urdf_bodies_after_merge()) == set(_preset_literal("BODY_NAMES"))


def test_num_bodies_field_matches_body_names():
    assert _preset_robot_field("num_bodies") == len(_preset_literal("BODY_NAMES"))


def test_key_bodies_exist_in_both_assets():
    key = _preset_robot_field("key_bodies")
    assert key, "key_bodies must not be empty"
    for name in key:
        assert name in _mjcf_bodies(), f"{name} missing from MJCF"
        assert name in _urdf_bodies_after_merge(), f"{name} merged away in URDF import"


def test_actuated_joint_sets_agree():
    m = mujoco.MjModel.from_xml_path(str(XML))
    hinges = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    }
    root = ET.parse(URDF).getroot()
    revolute = {j.get("name") for j in root.iter("joint") if j.get("type") == "revolute"}
    # the manifest's control set is the canonical 29 (test_manifest ties the
    # preset's limit arrays to it)
    from everest_locomotion.robots.manifest import load_manifest

    assert hinges == revolute == set(load_manifest("a3_ultra").control_joints)


def test_total_mass_agrees_between_assets():
    m = mujoco.MjModel.from_xml_path(str(XML))
    mjcf_mass = float(mujoco.mj_getTotalmass(m))
    root = ET.parse(URDF).getroot()
    urdf_mass = sum(
        float(link.find("inertial").find("mass").get("value"))
        for link in root.iter("link")
        if link.find("inertial") is not None
    )
    assert abs(mjcf_mass - urdf_mass) < 0.02
    assert abs(mjcf_mass - 60.1776) < 0.01


def test_collision_primitive_counts_agree():
    """Both engines must see the same contact geometry, not hulls vs boxes."""
    m = mujoco.MjModel.from_xml_path(str(XML))
    mjcf_collisions = sum(
        1
        for g in range(m.ngeom)
        if (m.geom_contype[g] or m.geom_conaffinity[g])
        and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) != "world"
    )
    root = ET.parse(URDF).getroot()
    urdf_collisions = sum(len(link.findall("collision")) for link in root.iter("link"))
    assert mjcf_collisions == urdf_collisions
    meshes = [
        c
        for link in root.iter("link")
        for c in link.findall("collision")
        if c.find("geometry/mesh") is not None
    ]
    assert not meshes, "URDF collisions must be primitives, not meshes"
