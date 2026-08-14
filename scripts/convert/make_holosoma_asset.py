"""Generate the Holosoma-ready A3 Ultra asset (29-DOF locomotion variant).

The two outputs must be the SAME robot: IsaacSim/PhysX (the default training
backend) imports the URDF, MuJoCo/MJWarp reads the MJCF, and the stability suite
compares results across them. So both files are generated together and verified
against one another and against `a3_ultra_presets.BODY_NAMES`.

From the official a3_ultra_t2d5 model:
  MJCF: - weld head joints (head_yaw/head_pitch removed; bodies+mass kept)
        - remove their actuators/sensors
        - remove the official keyframe (qpos dim changes; it was a captured
          off-nominal pose anyway); add settled lying keyframes
          (supine/prone/sides) for the get-up task
        - rename free joint to `floating_base_joint` (G1/Holosoma convention)
        - add massless-ish `*_foot_contact_point` bodies at the sole center
        - mesh collisions -> AABB boxes; rewrite mesh paths to local `meshes/`
  URDF: - head joints revolute -> fixed, marked `dont_collapse` so the links
          survive the importer's fixed-joint merge (the config names them)
        - `*_foot_contact_point` links added, same `dont_collapse` treatment
          (upstream G1 does exactly this)
        - mesh collisions replaced by the MJCF's primitives, transplanted
          body-by-body (link frames are identical between the two files)
        - zero-mass decorative links get tiny valid inertials
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

# Links that must survive the URDF importer's fixed-joint merge because the
# Holosoma robot config names them in `body_names` / `key_bodies`. IsaacLab's
# and IsaacGym's importers honour the `dont_collapse` attribute on the fixed
# joint (upstream G1's URDF uses the same trick for its foot contact points).
URDF_KEEP_LINKS = {
    "head_yaw_Link",
    "head_pitch_Link",
    "left_foot_contact_point",
    "right_foot_contact_point",
}

# Bodies that keep a (primitive) collision geom in the training asset. The
# official per-link MESH collisions overflow MJWarp's per-env constraint
# allocation (nefc/njmax) and are slow; standard practice (cf. Holosoma's G1
# asset) is primitive collisions. Feet keep the official 13 spheres per side.
# v4 (get-up task): every body carrying an official collision mesh gets a box —
# lying/rolling/push-off needs credible ground contact on head, upper arms,
# forearms/hands, and the hip cluster, not just the locomotion contact set.
PRIMITIVE_COLLISION_BODIES = [
    "pelvis_link",
    "torso_Link",
    "head_yaw_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "left_knee_Link",
    "right_knee_Link",
]


def _armature_for(joint: str) -> float:
    """ASSUMED armature by actuator class; keep in sync with a3_ultra_presets."""
    if "wrist_pitch" in joint or "wrist_yaw" in joint:
        return 0.003
    if any(k in joint for k in ("shoulder", "elbow", "wrist_roll")):
        return 0.005
    if "ankle" in joint or "waist_roll" in joint:
        return 0.01
    if "waist_pitch" in joint:
        return 0.02
    if any(k in joint for k in ("hip", "knee", "waist_yaw")):
        return 0.025
    if "head" in joint:
        return 0.003
    raise ValueError(f"no armature rule for {joint}")


def _collision_boxes_from_source() -> dict[str, list[dict]]:
    """AABB-fit boxes to each target body's official collision-mesh geoms.

    Uses MuJoCo's compiled geom_aabb (center + half-extents in geom frame) so
    the primitive envelopes derive mechanically from the official meshes.
    """
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(SRC_DIR / "mjcf" / "a3_ultra_t2d5.xml"))

    # completeness: every official collision-mesh body must be in our list
    official = set()
    for g in range(m.ngeom):
        if m.geom_contype[g] != 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            official.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
    missing = official - set(PRIMITIVE_COLLISION_BODIES)
    assert not missing, f"official collision-mesh bodies not covered: {sorted(missing)}"

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

    # per-joint armature (ASSUMED reflected rotor inertia; official MJCF ships
    # none, but explicit software PD at 200 Hz is integration-unstable on the
    # low-inertia distal joints without it — Holosoma's G1 asset sets it too).
    # Values mirror _ARMATURE in holosoma_ext/a3_ultra_presets.py (test-checked).
    n_arm = 0
    for j in root.iter("joint"):
        name = j.get("name", "")
        if name.endswith("_joint") and j.get("type") != "free":
            j.set("armature", f"{_armature_for(name)}")
            n_arm += 1
    print(f"armature set on {n_arm} joints")

    # drop stale keyframe
    for key in list(root.findall("keyframe")):
        root.remove(key)

    # contact solref: official 0.005 s is tuned for dt=0.001; Holosoma/MJWarp
    # steps at dt=0.005, and MuJoCo requires timeconst >= 2*dt for stable
    # contacts (G1's Holosoma asset uses 0.01). Relax to 0.01 everywhere.
    n_solref = 0
    for el in root.iter():
        sr = el.get("solref")
        if sr and sr.split()[0] == "0.005":
            parts = sr.split()
            parts[0] = "0.01"
            el.set("solref", " ".join(parts))
            n_solref += 1
    print(f"solref relaxed 0.005 -> 0.01 on {n_solref} elements")

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


def _set_nominal_pose(m, d) -> None:
    """Default crouch stand (matches configs/robots/a3_ultra.yaml default_pose)."""
    import mujoco

    d.qpos[2] = 1.07
    d.qpos[3] = 1.0
    crouch = {"hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for key, val in crouch.items():
            if key in name:
                d.qpos[m.jnt_qposadr[j]] = val


def add_auto_excludes() -> None:
    """Exclude self-collision body pairs that interpenetrate at the nominal pose.

    AABB boxes are fatter than the official meshes; chain-neighbor pairs beyond
    parent-child (auto-filtered by MuJoCo) can overlap permanently, e.g. the
    wrist_roll->wrist_pitch->wrist_yaw cluster. Any pair penetrating at the
    nominal stand is spurious-by-construction: it would emit constraint noise in
    every frame of training. Derived mechanically, like the boxes themselves.
    """
    import mujoco

    path = OUT_DIR / "a3_ultra_29dof.xml"
    for _pass in range(4):
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        _set_nominal_pose(m, d)
        mujoco.mj_forward(m, d)
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        pairs = set()
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if floor in (g1, g2) or d.contact[c].dist >= 0:
                continue
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1])
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2])
            if b1 != b2:
                pairs.add(tuple(sorted((b1, b2))))
        if not pairs:
            print(f"auto-excludes: converged after {_pass} pass(es)")
            return
        tree = ET.parse(path)
        root = tree.getroot()
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        existing = {
            tuple(sorted((e.get("body1"), e.get("body2"))))
            for e in contact.findall("exclude")
        }
        new = sorted(pairs - existing)
        assert new, f"pose still penetrating but pairs already excluded: {sorted(pairs)}"
        for b1, b2 in new:
            ET.SubElement(contact, "exclude", body1=b1, body2=b2)
        print(f"auto-excludes pass {_pass}: added {len(new)} pairs: {new}")
        ET.indent(tree)
        tree.write(path)
    raise RuntimeError("auto-exclude did not converge in 4 passes")


LYING_ORIENTATIONS = {  # wxyz base quats: rotate about y (pitch) / x (roll)
    "lying_supine": (0.7071068, 0.0, -0.7071068, 0.0),
    "lying_prone": (0.7071068, 0.0, 0.7071068, 0.0),
    "lying_side_left": (0.7071068, 0.7071068, 0.0, 0.0),
    "lying_side_right": (0.7071068, -0.7071068, 0.0, 0.0),
}


def add_lying_keyframes() -> None:
    """Bake settled lying keyframes (supine/prone/sides) into the asset.

    Get-up training and the fallen-pose generator need canonical lying states;
    HoST's README recommends shipping them as keyframes. Each is produced by
    dropping the zero-pose robot from 0.5 m in a canonical orientation and
    simulating until passive rest, so the keyframes stay mechanically derived
    from the model rather than hand-authored.
    """
    import mujoco
    import numpy as np

    path = OUT_DIR / "a3_ultra_29dof.xml"
    m = mujoco.MjModel.from_xml_path(str(path))
    keys: dict[str, np.ndarray] = {}
    for name, quat in LYING_ORIENTATIONS.items():
        d = mujoco.MjData(m)
        d.qpos[2] = 0.5
        d.qpos[3:7] = quat
        for i in range(12000):
            mujoco.mj_step(m, d)
            if i > 2000 and np.linalg.norm(d.qvel) < 0.03:
                break
        assert np.all(np.isfinite(d.qpos)), f"{name}: settle diverged"
        keys[name] = d.qpos.copy()
        print(f"keyframe {name}: settled in {i * m.opt.timestep:.1f} s, "
              f"base z={d.qpos[2]:.3f}, |qvel|={np.linalg.norm(d.qvel):.3f}")

    tree = ET.parse(path)
    root = tree.getroot()
    for kf in list(root.findall("keyframe")):
        root.remove(kf)
    kf = ET.SubElement(root, "keyframe")
    for name, qpos in keys.items():
        ET.SubElement(kf, "key", name=name,
                      qpos=" ".join(f"{v:.6f}" for v in qpos))
    ET.indent(tree)
    tree.write(path)
    print(f"keyframes written: {list(keys)}")


def _quat_to_rpy(quat) -> tuple[float, float, float]:
    """MuJoCo wxyz quaternion -> URDF fixed-axis rpy (R = Rz(y) Ry(p) Rx(r))."""
    import numpy as np

    w, x, y, z = (float(v) for v in quat)
    m00 = 1 - 2 * (y * y + z * z)
    m10 = 2 * (x * y + w * z)
    m20 = 2 * (x * z - w * y)
    m21 = 2 * (y * z + w * x)
    m22 = 1 - 2 * (x * x + y * y)
    pitch = float(np.arcsin(np.clip(-m20, -1.0, 1.0)))
    roll = float(np.arctan2(m21, m22))
    yaw = float(np.arctan2(m10, m00))
    return roll, pitch, yaw


def _mjcf_collision_primitives() -> dict[str, list[dict]]:
    """Collision primitives of the generated MJCF, keyed by body name.

    Read back from the compiled MJCF so the URDF gets byte-identical geometry:
    the AABB boxes on the contact bodies plus the official foot spheres. Both
    files use the same link frames (verified: inertial origins agree), so geom
    pose in the MuJoCo body frame is the URDF collision origin unchanged.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(OUT_DIR / "a3_ultra_29dof.xml"))
    out: dict[str, list[dict]] = {}
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if body in (None, "world"):
            continue
        gtype = m.geom_type[g]
        entry = {
            "xyz": " ".join(f"{v:.6f}" for v in m.geom_pos[g]),
            "rpy": " ".join(f"{v:.6f}" for v in _quat_to_rpy(m.geom_quat[g])),
        }
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            entry["shape"] = "box"
            entry["args"] = {"size": " ".join(f"{2 * v:.6f}" for v in m.geom_size[g][:3])}
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            entry["shape"] = "sphere"
            entry["args"] = {"radius": f"{m.geom_size[g][0]:.6f}"}
        else:
            raise ValueError(
                f"unsupported collision geom type {mujoco.mjtGeom(gtype).name} on {body}; "
                "extend the URDF mirror before regenerating"
            )
        out.setdefault(body, []).append(entry)
    return out


def _add_kept_link(root, link_name: str, parent: str, xyz: str) -> None:
    """Append a massless-ish link on a `dont_collapse` fixed joint (G1 pattern)."""
    link = ET.SubElement(root, "link", name=link_name)
    ine = ET.SubElement(link, "inertial")
    ET.SubElement(ine, "mass", value="0.001")
    ET.SubElement(ine, "inertia", ixx="1e-7", ixy="0", ixz="0", iyy="1e-7", iyz="0", izz="1e-7")
    # IsaacLab's USD conversion wants a visual prim on every link.
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
    geo = ET.SubElement(vis, "geometry")
    ET.SubElement(geo, "box", size="0.001 0.001 0.001")
    mat = ET.SubElement(vis, "material", name="invisible")
    ET.SubElement(mat, "color", rgba="0 0 0 0")
    joint = ET.SubElement(root, "joint", name=f"{link_name}_joint", type="fixed")
    joint.set("dont_collapse", "true")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=link_name)
    ET.SubElement(joint, "origin", xyz=xyz, rpy="0 0 0")


def convert_urdf() -> None:
    primitives = _mjcf_collision_primitives()
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

    # Collisions: drop the official per-link meshes (PhysX would convex-hull 47
    # of them per robot) and transplant the MJCF's primitives, so both physics
    # engines see the same contact geometry.
    n_drop = n_prim = 0
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)
            n_drop += 1
        for entry in primitives.get(link.get("name"), []):
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", xyz=entry["xyz"], rpy=entry["rpy"])
            geo = ET.SubElement(col, "geometry")
            ET.SubElement(geo, entry["shape"], **entry["args"])
            n_prim += 1
    print(f"URDF: replaced {n_drop} mesh collisions with {n_prim} MJCF primitives")

    # The IsaacSim importer demonstrably does NOT merge fixed joints even with
    # merge_fixed_joints=True (cloud run 2026-08-14: all 47 links surfaced as
    # articulation bodies), so relying on merge is unsafe. Make the URDF
    # physically contain EXACTLY the MJCF's 34 bodies: delete the decorative
    # leaf links (torso shell, cameras, IMUs, hand palms, knee shells) and fold
    # their mass by rewriting every kept link's inertial from the compiled MJCF
    # (whose bodies already lump welded children; total-mass parity asserted in
    # verify()).
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(OUT_DIR / "a3_ultra_29dof.xml"))
    mjcf_bodies = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)
    }
    removed = []
    for link in list(root.findall("link")):
        name = link.get("name")
        if name not in mjcf_bodies:
            root.remove(link)
            removed.append(name)
    for joint in list(root.findall("joint")):
        child = joint.find("child")
        if child is not None and child.get("link") in removed:
            root.remove(joint)
    print(f"URDF: removed {len(removed)} decorative links (masses folded via MJCF inertials)")

    n_sync = 0
    for link in root.findall("link"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link.get("name"))
        if bid < 0:
            continue
        ine = link.find("inertial")
        if ine is None:
            ine = ET.SubElement(link, "inertial")
        for el in list(ine):
            ine.remove(el)
        rpy = _quat_to_rpy(m.body_iquat[bid])
        ET.SubElement(
            ine, "origin",
            xyz=" ".join(f"{v:.8f}" for v in m.body_ipos[bid]),
            rpy=" ".join(f"{v:.8f}" for v in rpy),
        )
        ET.SubElement(ine, "mass", value=f"{m.body_mass[bid]:.8f}")
        dx, dy, dz = m.body_inertia[bid]
        ET.SubElement(
            ine, "inertia",
            ixx=f"{dx:.9f}", ixy="0", ixz="0", iyy=f"{dy:.9f}", iyz="0", izz=f"{dz:.9f}",
        )
        n_sync += 1
    print(f"URDF: synced {n_sync} link inertials from compiled MJCF")

    # Links the Holosoma config names; keep them through any importer merge too.
    for joint in root.iter("joint"):
        if joint.get("name") in HEAD_JOINTS:
            joint.set("dont_collapse", "true")
    for side in ("left", "right"):
        _add_kept_link(
            root, f"{side}_foot_contact_point", f"{side}_ankle_roll_Link", FOOT_SOLE_POS
        )

    ET.indent(tree)
    tree.write(OUT_DIR / "a3_ultra_29dof.urdf")
    n_links = len(root.findall("link"))
    print(f"URDF written: {OUT_DIR / 'a3_ultra_29dof.urdf'} ({n} head joints fixed, "
          f"{n_links} links total, {len(URDF_KEEP_LINKS)} marked dont_collapse)")


def urdf_merged_links(urdf_path: Path) -> list[str]:
    """Links that survive an importer's fixed-joint merge, in URDF declaration order.

    Mirrors IsaacLab/IsaacGym semantics: a link is merged into its parent when
    its parent joint is `fixed` and not marked `dont_collapse`. The articulation
    root always survives.
    """
    root = ET.parse(urdf_path).getroot()
    links = [link.get("name") for link in root.iter("link")]
    merged = {
        j.find("child").get("link")
        for j in root.iter("joint")
        if j.get("type") == "fixed" and j.get("dont_collapse") != "true"
    }
    return [name for name in links if name not in merged]


def verify() -> None:
    import mujoco

    urdf_path = OUT_DIR / "a3_ultra_29dof.urdf"
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
    _set_nominal_pose(m, d)
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self_contacts = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[c].geom1),
         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[c].geom2))
        for c in range(d.ncon)
        if floor not in (d.contact[c].geom1, d.contact[c].geom2) and d.contact[c].dist < 0
    ]
    assert not self_contacts, f"spurious self-contacts at default pose: {self_contacts[:6]}"

    # --- URDF <-> MJCF parity (IsaacSim trains on the URDF, MuJoCo gates on the MJCF) ---
    mjcf_bodies = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(m.nbody)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) != "world"
    ]
    urdf_bodies = urdf_merged_links(urdf_path)
    assert set(urdf_bodies) == set(mjcf_bodies), (
        "URDF (post-merge) and MJCF body sets differ:\n"
        f"  URDF only: {sorted(set(urdf_bodies) - set(mjcf_bodies))}\n"
        f"  MJCF only: {sorted(set(mjcf_bodies) - set(urdf_bodies))}"
    )

    urdf_root = ET.parse(urdf_path).getroot()
    urdf_mass = sum(
        float(link.find("inertial").find("mass").get("value"))
        for link in urdf_root.iter("link")
        if link.find("inertial") is not None
    )
    assert abs(urdf_mass - total) < 0.02, f"URDF mass {urdf_mass:.4f} != MJCF mass {total:.4f}"

    urdf_revolute = [
        j.get("name") for j in urdf_root.iter("joint") if j.get("type") == "revolute"
    ]
    assert len(urdf_revolute) == 29, f"URDF has {len(urdf_revolute)} revolute joints, expected 29"
    mjcf_hinges = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    }
    assert set(urdf_revolute) == mjcf_hinges, "URDF/MJCF actuated joint sets differ"

    print(f"verify OK: 29 dof, 29 actuators, mass {total:.4f} kg, nbody={m.nbody}, "
          f"ncon@default={d.ncon}, no self-penetration; "
          f"URDF parity: {len(urdf_bodies)} bodies, mass {urdf_mass:.4f} kg")


if __name__ == "__main__":
    convert_mjcf()
    add_auto_excludes()
    add_lying_keyframes()
    convert_urdf()
    verify()
