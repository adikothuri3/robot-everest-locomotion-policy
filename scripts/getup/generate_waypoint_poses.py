"""Generate waypoint (KSI) poses for get-up training: sit / kneel / crouch.

Key-State Initialization: resetting a fraction of envs to states ALONG the
get-up route densifies the hard sit->crouch->rise region. Evidence: HiFAR
resets from 6 keyframes on handcrafted recovery motions (hardware-proven),
UniReLo's init-distribution ablation is its largest (up to 24 pts), and the
H1-2 recovery paper (arXiv:2603.08619) initializes from exactly these
categories (sitting/kneeling/squatting) plus lying.

Each seed pose is perturbed, settled briefly under stiff PD hold toward the
seed (a free drop would collapse these non-passive poses), then saved if it
stayed in-family. Output: assets/a3_ultra/getup/fallen_poses_waypoint.npy
(same layout as the fallen banks) + meta update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.xml"
OUT_DIR = REPO / "assets" / "a3_ultra" / "getup"

# canonical seeds: {joint substring pattern: value}, base z, description.
# Values chosen inside official limits (knee max 2.53, hip pitch -2.51..2.93,
# ankle pitch -0.91..0.52).
SEEDS = {
    "sit": (
        {"hip_pitch": -1.55, "knee": 0.35, "ankle_pitch": 0.1,
         "shoulder_pitch": -0.3, "elbow": 0.5},
        0.17,
    ),
    "kneel": (
        {"hip_pitch": -0.35, "knee": 2.35, "ankle_pitch": 0.45,
         "shoulder_pitch": 0.0, "elbow": 0.3},
        0.52,
    ),
    "crouch": (
        {"hip_pitch": -2.1, "knee": 2.35, "ankle_pitch": -0.5,
         "shoulder_pitch": -1.0, "elbow": 0.5},
        0.42,
    ),
}
HOLD_KP = {"hip": 300.0, "knee": 300.0, "ankle": 150.0, "waist": 300.0,
           "shoulder": 80.0, "elbow": 80.0, "wrist": 15.0}
HOLD_KD_FRac = 0.1


def _kp_for(name: str) -> float:
    for k, v in HOLD_KP.items():
        if k in name:
            return v
    return 50.0


def generate(num_per_seed: int, seed: int) -> dict:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    rng = np.random.default_rng(seed)
    hinge = [(j, m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j, _, _ in hinge]
    kp = np.array([_kp_for(n) for n in names])
    kd = kp * HOLD_KD_FRac
    act_joint = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                                   m.actuator_trnid[a, 0]) for a in range(m.nu)]

    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    kept: dict[str, list[np.ndarray]] = {k: [] for k in SEEDS}
    for cat, (pattern, base_z) in SEEDS.items():
        target = np.zeros(len(hinge))
        for i, n in enumerate(names):
            for key, val in pattern.items():
                if key in n:
                    target[i] = -val if (n.startswith("right") and key == "shoulder_roll") else val
        tries = 0
        while len(kept[cat]) < num_per_seed and tries < num_per_seed * 6:
            tries += 1
            d = mujoco.MjData(m)
            q = target + rng.uniform(-0.08, 0.08, len(hinge))
            lo = m.jnt_range[[j for j, _, _ in hinge], 0]
            hi = m.jnt_range[[j for j, _, _ in hinge], 1]
            q = np.clip(q, lo, hi)
            yaw = rng.uniform(-np.pi, np.pi)
            pitch = rng.uniform(-0.15, 0.15)
            qy = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
            qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, qz, qy)
            d.qpos[3:7] = quat
            for (j, qadr, _), v in zip(hinge, q):
                d.qpos[qadr] = v
            # spawn height: scan upward for the lowest penetration-free z
            # (never trust a nominal height — sim2sim lesson)
            d.qpos[2] = base_z - 0.15
            for _ in range(60):
                mujoco.mj_forward(m, d)
                # ONLY floor contacts count: self-contact penetration (arm box
                # vs torso etc.) is z-independent and would push the scan to
                # its cap, spawning the robot airborne
                floor_pen = max(
                    (-d.contact[c].dist for c in range(d.ncon)
                     if floor_gid in (d.contact[c].geom1, d.contact[c].geom2)),
                    default=0.0,
                )
                if floor_pen < 0.002:
                    break
                d.qpos[2] += 0.01
            # SHORT settle (0.3 s): resolve contacts, don't demand equilibrium —
            # KSI keyframes are mid-motion states (HiFAR uses motion frames);
            # the policy's job is to catch them
            hold = {n: v for n, v in zip(names, q)}
            for _ in range(300):
                for a in range(m.nu):
                    n = act_joint[a]
                    i = names.index(n)
                    j, qadr, dadr = hinge[i]
                    d.ctrl[a] = kp[i] * (hold[n] - d.qpos[qadr]) - kd[i] * d.qvel[dadr]
                mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                continue
            # in-family filter: not yet toppled, plausible height band
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, d.qpos[3:7])
            pg_z = (rot.reshape(3, 3).T @ np.array([0, 0, -1.0]))[2]
            h = d.qpos[2]
            if np.linalg.norm(d.qvel) < 2.5 and pg_z < -0.45 and 0.10 < h < 0.80:
                d.qvel[:] = 0.0
                kept[cat].append(d.qpos.copy())
    rows = [p for cat in SEEDS for p in kept[cat]]
    arr = np.array(rows) if rows else np.zeros((0, m.nq))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "fallen_poses_waypoint.npy", arr)
    meta_path = OUT_DIR / "fallen_poses_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["counts"]["waypoint"] = len(arr)
    meta["waypoint_seeds"] = {k: len(v) for k, v in kept.items()}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"kept {({k: len(v) for k, v in kept.items()})} -> {len(arr)} waypoint poses")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-per-seed", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate(a.num_per_seed, a.seed)
