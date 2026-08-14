"""Generate a fallen-pose bank for A3 Ultra get-up training.

Method (HumanUP, arXiv:2502.12152 — their script is not public): start from a
canonical lying keyframe, randomize joint positions around it, drop from 0.5 m
with random yaw, simulate passively until rest, keep the settled state. Poses
are classified by final orientation (projected gravity matched against the
canonical lying keyframes), not by the orientation they were dropped in — a
drop that rolls from supine to side is stored as side.

Output (default assets/a3_ultra/getup/):
  fallen_poses_{supine,prone,side_left,side_right}.npy  float64 (N_cat, nq)
  fallen_poses_meta.json                                generation record

Usage:
  python scripts/getup/generate_fallen_poses.py            # 2048 poses
  python scripts/getup/generate_fallen_poses.py --num 128 --seed 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.xml"
DEFAULT_OUT = REPO / "assets" / "a3_ultra" / "getup"

KEYFRAMES = ["lying_supine", "lying_prone", "lying_side_left", "lying_side_right"]
CATEGORIES = ["supine", "prone", "side_left", "side_right"]

DROP_HEIGHT = 0.5
RAND_FRAC = 0.35          # joint randomization: fraction of each joint's range
SETTLE_MAX_S = 10.0
SETTLE_MIN_S = 2.0
SETTLE_QVEL = 0.05


def _projected_gravity(qpos: np.ndarray) -> np.ndarray:
    """World -z expressed in the base frame (wxyz quat at qpos[3:7])."""
    w, x, y, z = qpos[3:7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.array([w, x, y, z]))
    return rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


def generate(model_path: Path, out_dir: Path, num: int, seed: int) -> dict:
    m = mujoco.MjModel.from_xml_path(str(model_path))
    rng = np.random.default_rng(seed)

    key_qpos = {}
    for name in KEYFRAMES:
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, name)
        assert kid >= 0, f"keyframe {name} missing — regenerate the asset"
        key_qpos[name] = m.key_qpos[kid].copy()
    ref_gravity = {
        cat: _projected_gravity(key_qpos[f"lying_{cat}"]) for cat in CATEGORIES
    }

    jnt_range = m.jnt_range.copy()
    hinge_adr = [
        m.jnt_qposadr[j]
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    hinge_rng = np.array(
        [jnt_range[j] for j in range(m.njnt)
         if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    )

    banks: dict[str, list[np.ndarray]] = {cat: [] for cat in CATEGORIES}
    n_unsettled = 0
    d = mujoco.MjData(m)
    max_steps = int(SETTLE_MAX_S / m.opt.timestep)
    min_steps = int(SETTLE_MIN_S / m.opt.timestep)
    t0 = time.time()

    for i in range(num):
        base = key_qpos[KEYFRAMES[i % len(KEYFRAMES)]].copy()
        mujoco.mj_resetData(m, d)
        d.qpos[:] = base
        d.qpos[2] = DROP_HEIGHT
        # random yaw about world z, composed onto the lying orientation
        yaw = rng.uniform(-np.pi, np.pi)
        qyaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        qnew = np.zeros(4)
        mujoco.mju_mulQuat(qnew, qyaw, d.qpos[3:7])
        d.qpos[3:7] = qnew
        # randomize joints around the settled keyframe pose, clipped to limits
        span = (hinge_rng[:, 1] - hinge_rng[:, 0]) * RAND_FRAC
        noise = rng.uniform(-0.5, 0.5, len(hinge_adr)) * span
        for k, adr in enumerate(hinge_adr):
            d.qpos[adr] = np.clip(
                d.qpos[adr] + noise[k], hinge_rng[k, 0], hinge_rng[k, 1]
            )
        for step in range(max_steps):
            mujoco.mj_step(m, d)
            if step > min_steps and np.linalg.norm(d.qvel) < SETTLE_QVEL:
                break
        if not np.all(np.isfinite(d.qpos)) or np.linalg.norm(d.qvel) > 0.5:
            n_unsettled += 1
            continue
        pg = _projected_gravity(d.qpos)
        cat = max(CATEGORIES, key=lambda c: float(pg @ ref_gravity[c]))
        banks[cat].append(d.qpos.copy())
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{num} ({time.time() - t0:.0f} s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cat in CATEGORIES:
        arr = np.array(banks[cat]) if banks[cat] else np.zeros((0, m.nq))
        np.save(out_dir / f"fallen_poses_{cat}.npy", arr)
        counts[cat] = len(arr)
    hinge_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    meta = {
        "model": str(model_path.relative_to(REPO)),
        "nq": int(m.nq),
        # MJCF kinematic-TREE order (waist subtree first) — NOT the canonical
        # actuator order from configs/robots/a3_ultra.yaml. Consumers MUST
        # remap by name. Base quat is MuJoCo wxyz.
        "hinge_joint_names_tree_order": hinge_names,
        "num_requested": num,
        "seed": seed,
        "drop_height_m": DROP_HEIGHT,
        "rand_frac_of_range": RAND_FRAC,
        "settle": {"max_s": SETTLE_MAX_S, "min_s": SETTLE_MIN_S,
                   "qvel_thresh": SETTLE_QVEL},
        "counts": counts,
        "unsettled_discarded": n_unsettled,
        "qpos_layout": "0:3 base pos, 3:7 base quat wxyz, 7: 29 hinge joints in MJCF tree order",
    }
    (out_dir / "fallen_poses_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {counts}, discarded {n_unsettled}, "
          f"wall {time.time() - t0:.0f} s -> {out_dir}")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--num", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.model, args.out, args.num, args.seed)
