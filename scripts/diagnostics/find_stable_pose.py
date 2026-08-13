"""Empirically find a stable A3 Ultra standing pose.

Tries candidate (hip_pitch, knee, ankle_pitch) crouches with both sign
conventions, computes COM vs foot support at qpos0, then settles 4 s under PD
hold and reports final uprightness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from everest_locomotion.robots.manifest import load_manifest
from everest_locomotion.sim_adapters.mujoco_adapter import MujocoRobot

manifest = load_manifest()
robot = MujocoRobot(manifest)
m, d = robot.model, robot.data

idx = {j: i for i, j in enumerate(manifest.joint_order)}

candidates = []
for hip in (0.0, -0.1, -0.2, -0.3, 0.1, 0.2, 0.3):
    for knee in (0.0, 0.2, 0.4, 0.6, -0.1):
        ankle = -(hip + knee)  # keep foot flat if axes compose about +y
        if not (-0.9 < ankle < 0.52):
            continue
        candidates.append((hip, knee, ankle))

results = []
for hip, knee, ankle in candidates:
    pose = manifest.default_pose_vector().copy()
    for side in ("left", "right"):
        pose[idx[f"{side}_hip_pitch_joint"]] = hip
        pose[idx[f"{side}_knee_joint"]] = knee
        pose[idx[f"{side}_ankle_pitch_joint"]] = ankle

    # FK at generous height to find foot-bottom offset, then reset at right height
    robot.reset_to_default(base_height=1.3)
    d.qpos[robot.qpos_adr] = pose
    mujoco.mj_forward(m, d)
    lowest = min(
        d.geom_xpos[g][2] - m.geom_rbound[g]
        for g in range(m.ngeom)
        if m.geom_bodyid[g] != 0 and m.geom_contype[g]
    )
    stand_h = 1.3 - lowest + 0.002
    com = d.subtree_com[robot.base_body_id].copy()
    foot_x = np.mean([d.xpos[b][0] for b in robot.feet_body_ids])

    robot.reset_to_default(base_height=stand_h)
    d.qpos[robot.qpos_adr] = pose
    mujoco.mj_forward(m, d)
    for _ in range(4000):
        robot.kp, robot.kd = manifest.pd_gains()
        tau = robot.kp * (pose - robot.joint_pos()) - robot.kd * robot.joint_vel()
        d.ctrl[robot.act_ids] = np.clip(tau, -robot.effort_limit, robot.effort_limit)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            break
    grav = robot.projected_gravity()
    forces = robot.foot_contact_forces()
    results.append((hip, knee, ankle, stand_h, com[0] - foot_x, grav[2], robot.base_pos()[2], sum(forces)))

print(f"{'hip':>6} {'knee':>6} {'ankle':>6} {'h0':>6} {'comdx':>7} {'gz':>7} {'z_end':>6} {'F_feet':>7}")
for r in sorted(results, key=lambda r: r[5]):
    print(f"{r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.3f} {r[4]:7.3f} {r[5]:7.3f} {r[6]:6.3f} {r[7]:7.0f}")
