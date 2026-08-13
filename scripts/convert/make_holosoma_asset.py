"""Generate the Holosoma-ready A3 Ultra asset (29-DOF locomotion variant).

From the official a3_ultra_t2d5 model:
  MJCF: - weld head joints (head_yaw/head_pitch removed; bodies+mass kept)
        - remove their actuators/sensors
        - remove the official keyframe (qpos dim changes; it was a captured
          off-nominal pose anyway)
        - rename free joint to `floating_base_joint` (G1/Holosoma convention)
        - add massless-ish `*_foot_contact_point` bodies at the sole center
        - rewrite mesh paths to local `meshes/`
  URDF: - head joints revolute -> fixed
  Copies referenced meshes next to the outputs.

Output: assets/a3_ultra/holosoma/{a3_ultra_29dof.xml, a3_ultra_29dof.urdf, meshes/}

Every transformation here is mechanical; no inertial/limit values are altered.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "third_party" / "A3-A3U-robot-model" / "a3_ultra_t2d5"
OUT_DIR = REPO / "assets" / "a3_ultra" / "holosoma"

HEAD_JOINTS = {"head_yaw_joint", "head_pitch_joint"}
FOOT_SOLE_POS = "0.04 0 -0.067"  # sole center in ankle_roll frame (= official foot site)


def convert_mjcf() -> None:
    tree = ET.parse(SRC_DIR / "mjcf" / "a3_ultra_t2d5.xml")
    root = tree.getroot()
    root.set("model", "a3_ultra_29dof")

    # mesh paths -> local meshes/
    used_meshes = []
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        name = Path(f).name
        mesh.set("file", f"meshes/{name}")
        used_meshes.append(name)

    # remove head joints (weld) and their sensors/actuators
    for parent in root.iter():
        for j in list(parent.findall("joint")):
            if j.get("name") in HEAD_JOINTS:
                parent.remove(j)
    actuator = root.find("actuator")
    for motor in list(actuator.findall("motor")):
        if motor.get("joint") in HEAD_JOINTS:
            actuator.remove(motor)
    sensors = root.find("sensor")
    if sensors is not None:
        for s in list(sensors):
            if s.get("joint") in HEAD_JOINTS:
                sensors.remove(s)

    # rename free joint
    for j in root.iter("joint"):
        if j.get("type") == "free":
            j.set("name", "floating_base_joint")

    # drop stale keyframe
    for key in list(root.findall("keyframe")):
        root.remove(key)

    # add foot contact point bodies
    for body in root.iter("body"):
        name = body.get("name", "")
        if name in ("left_ankle_roll_Link", "right_ankle_roll_Link"):
            side = "left" if name.startswith("left") else "right"
            cp = ET.SubElement(body, "body", name=f"{side}_foot_contact_point", pos=FOOT_SOLE_POS)
            ET.SubElement(cp, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-7 1e-7 1e-7")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(OUT_DIR / "a3_ultra_29dof.xml")

    mesh_out = OUT_DIR / "meshes"
    mesh_out.mkdir(exist_ok=True)
    for name in used_meshes:
        shutil.copy2(SRC_DIR / "meshes" / name, mesh_out / name)
    print(f"MJCF written: {OUT_DIR / 'a3_ultra_29dof.xml'} ({len(used_meshes)} meshes copied)")


def convert_urdf() -> None:
    tree = ET.parse(SRC_DIR / "urdf" / "model.urdf")
    root = tree.getroot()
    root.set("name", "a3_ultra_29dof")
    n = 0
    for joint in root.iter("joint"):
        if joint.get("name") in HEAD_JOINTS:
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                for el in list(joint.findall(tag)):
                    joint.remove(el)
            n += 1
    # mesh paths: point into local meshes/
    for mesh in root.iter("mesh"):
        f = mesh.get("filename")
        if f:
            mesh.set("filename", f"meshes/{Path(f).name}")
    ET.indent(tree)
    tree.write(OUT_DIR / "a3_ultra_29dof.urdf")
    print(f"URDF written: {OUT_DIR / 'a3_ultra_29dof.urdf'} ({n} head joints fixed)")


def verify() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(OUT_DIR / "a3_ultra_29dof.xml"))
    hinges = sum(1 for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    assert hinges == 29, f"expected 29 hinge joints, got {hinges}"
    assert m.nu == 29, f"expected 29 actuators, got {m.nu}"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_foot_contact_point") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint") >= 0
    total = mujoco.mj_getTotalmass(m)
    assert abs(total - 60.1776) < 0.01, f"mass changed: {total}"
    print(f"verify OK: 29 dof, 29 actuators, mass {total:.4f} kg, nbody={m.nbody}")


if __name__ == "__main__":
    convert_mjcf()
    convert_urdf()
    verify()
