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

# Bodies that keep a (primitive) collision geom in the training asset. The
# official per-link MESH collisions overflow MJWarp's per-env constraint
# allocation (nefc/njmax) and are slow; standard practice (cf. Holosoma's G1
# asset) is primitive collisions. Feet keep the official 13 spheres per side.
# These bodies cover termination/penalty contacts (pelvis/shoulder/hip) plus
# common fall contacts (torso, thigh, shin, elbow).
PRIMITIVE_COLLISION_BODIES = [
    "pelvis_link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "left_knee_Link",
    "right_knee_Link",
    "left_elbow_Link",
    "right_elbow_Link",
]


def _collision_boxes_from_source() -> dict[str, list[dict]]:
    """AABB-fit boxes to each target body's official collision-mesh geoms.

    Uses MuJoCo's compiled geom_aabb (center + half-extents in geom frame) so
    the primitive envelopes derive mechanically from the official meshes.
    """
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(SRC_DIR / "mjcf" / "a3_ultra_t2d5.xml"))
    boxes: dict[str, list[dict]] = {}
    for body in PRIMITIVE_COLLISION_BODIES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        assert bid >= 0, f"body {body} not found in source model"
        entries = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_contype[g] == 0:
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            center = m.geom_aabb[g][:3]
            half = np.maximum(m.geom_aabb[g][3:], 0.005)
            quat = m.geom_quat[g]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            pos = m.geom_pos[g] + rot.reshape(3, 3) @ center
            entries.append(
                {
                    "pos": " ".join(f"{v:.6f}" for v in pos),
                    "quat": " ".join(f"{v:.6f}" for v in quat),
                    "size": " ".join(f"{v:.6f}" for v in half),
                }
            )
        assert entries, f"no collision mesh found on {body}"
        boxes[body] = entries
    return boxes


def convert_mjcf() -> None:
    collision_boxes = _collision_boxes_from_source()
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

    # collision simplification: drop ALL mesh collision geoms; add AABB boxes
    # on termination/contact bodies (feet keep their official sphere geoms).
    n_removed = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("class") == "collision" and geom.get("type") == "mesh":
                body.remove(geom)
                n_removed += 1
        name = body.get("name", "")
        if name in collision_boxes:
            for i, bx in enumerate(collision_boxes[name]):
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{name}_collision_box{i}",
                    **{"class": "collision"},
                    type="box",
                    size=bx["size"],
                    pos=bx["pos"],
                    quat=bx["quat"],
                )
    print(f"collision simplification: removed {n_removed} mesh collision geoms, "
          f"added {sum(len(v) for v in collision_boxes.values())} boxes")

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

    # The official URDF gives decorative shell/sensor links zero mass AND zero
    # inertia; URDF importers (Isaac Sim) then auto-compute mass from mesh
    # volume (the torso shell alone becomes ~18.6 kg). Stamp tiny valid
    # inertials so importers keep the official 60.18 kg total.
    n_zero = 0
    for link in root.iter("link"):
        ine = link.find("inertial")
        if ine is None:
            continue
        if float(ine.find("mass").get("value")) == 0.0:
            ine.find("mass").set("value", "1e-4")
            inertia = ine.find("inertia")
            for k in ("ixx", "iyy", "izz"):
                inertia.set(k, "1e-7")
            for k in ("ixy", "ixz", "iyz"):
                inertia.set(k, "0")
            n_zero += 1
    print(f"URDF: stamped tiny inertials on {n_zero} zero-mass links")
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

    # no spurious self-contacts at the default crouch pose
    d = mujoco.MjData(m)
    d.qpos[2] = 1.07
    d.qpos[3] = 1.0
    crouch = {"hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for key, val in crouch.items():
            if key in name:
                d.qpos[m.jnt_qposadr[j]] = val
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self_contacts = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[c].geom1),
         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[c].geom2))
        for c in range(d.ncon)
        if floor not in (d.contact[c].geom1, d.contact[c].geom2) and d.contact[c].dist < 0
    ]
    assert not self_contacts, f"spurious self-contacts at default pose: {self_contacts[:6]}"
    print(f"verify OK: 29 dof, 29 actuators, mass {total:.4f} kg, nbody={m.nbody}, "
          f"ncon@default={d.ncon}, no self-penetration")


if __name__ == "__main__":
    convert_mjcf()
    convert_urdf()
    verify()
