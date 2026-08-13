"""A3 Ultra MuJoCo model diagnostics — prints PASS/FAIL for each check.

Checks:
  1. model loads
  2. manifest joint names/count match MJCF
  3. actuator mapping matches manifest order
  4. inertial sanity (positive masses/inertias, triangle inequality)
  5. passive settle: robot dropped in default pose under PD hold must not
     fall, explode, or produce NaNs
  6. both feet in contact with ground after settle
  7. no deep penetrations
  8. actuator ranges sane (torque limits from joints, not degenerate)
  9. short commanded rollout (hold pose) stays upright
 10. optional: render rollout video (--render, needs working GL)

Usage:
    python scripts/diagnostics/check_mujoco_model.py [--render]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from everest_locomotion.robots.manifest import load_manifest
from everest_locomotion.sim_adapters.mujoco_adapter import MujocoRobot

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="save settle rollout video")
    parser.add_argument("--robot", default="a3_ultra")
    args = parser.parse_args()

    manifest = load_manifest(args.robot)

    # 1. load
    try:
        robot = MujocoRobot(manifest)
        check("model loads", True, f"{manifest.mjcf_path.name}")
    except Exception as e:  # noqa: BLE001
        check("model loads", False, str(e))
        return finish()

    m, d = robot.model, robot.data

    # 2. joints
    mjcf_hinges = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    check(
        "manifest joints match MJCF",
        sorted(mjcf_hinges) == sorted(manifest.joint_order),
        f"{len(mjcf_hinges)} hinge joints",
    )

    # 3. actuator mapping: actuator i must drive the joint it is named after
    act_ok = True
    for name, aid in zip(manifest.joint_order, robot.act_ids):
        jid = m.actuator_trnid[aid, 0]
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) != name:
            act_ok = False
    check("actuator->joint mapping consistent", act_ok, f"{m.nu} actuators")

    # 4. inertial sanity
    masses = m.body_mass[1:]
    inertias = m.body_inertia[1:]
    tri = np.all(inertias.sum(axis=1) >= 2 * inertias.max(axis=1) - 1e-9)
    check(
        "inertials valid",
        bool((masses > 0).all() and (inertias > 0).all() and tri),
        f"total mass {mujoco.mj_getTotalmass(m):.2f} kg",
    )
    check(
        "total mass matches manifest",
        abs(mujoco.mj_getTotalmass(m) - manifest.total_mass_kg) < 0.01,
        f"{mujoco.mj_getTotalmass(m):.4f} vs {manifest.total_mass_kg}",
    )

    # 5. settle under stiff PD hold, 3 s.
    # Hold-mode gains: passive standing needs ankle stiffness > m*g*h_com
    # (~530 Nm/rad here) — RL gains are deliberately much softer, so this
    # diagnostic stiffens the hold. It validates the MODEL (contacts,
    # inertias, actuator mapping), not a realistic controller.
    hold_min = {"ankle": (800.0, 20.0), "hip": (400.0, 10.0), "knee": (400.0, 10.0), "waist": (400.0, 10.0)}
    for i, jname in enumerate(manifest.joint_order):
        for key, (kp, kd) in hold_min.items():
            if key in jname:
                robot.kp[i] = max(robot.kp[i], kp)
                robot.kd[i] = max(robot.kd[i], kd)
    robot.reset_to_default()
    target = robot.default_pose.copy()
    n_steps = int(3.0 / m.opt.timestep)
    frames = []
    renderer = None
    if args.render:
        try:
            renderer = mujoco.Renderer(m, height=480, width=640)
        except Exception as e:  # noqa: BLE001
            print(f"  (render disabled: {e})")
    exploded = False
    for i in range(n_steps):
        robot.apply_pd(target)
        robot.step()
        if not np.isfinite(d.qpos).all() or abs(d.qvel).max() > 100.0:
            exploded = True
            break
        if renderer and i % 40 == 0:
            renderer.update_scene(d)
            frames.append(renderer.render())
    base_z = robot.base_pos()[2]
    grav = robot.projected_gravity()
    upright = grav[2] < -0.95  # base z-axis within ~18 deg of vertical
    check("settle: no NaN/explosion", not exploded, f"max |qvel|={abs(d.qvel).max():.2f}")
    check(
        "settle: robot upright and at height",
        bool(upright and 0.9 < base_z < 1.2),
        f"base_z={base_z:.3f} m, gravity_z_in_base={grav[2]:.3f}",
    )

    # 6. feet contact
    forces = robot.foot_contact_forces()
    weight = mujoco.mj_getTotalmass(m) * 9.81
    check(
        "both feet in ground contact",
        all(f > 20.0 for f in forces),
        f"L={forces[0]:.0f} N R={forces[1]:.0f} N (weight {weight:.0f} N)",
    )

    # 7. penetration depth
    max_pen = 0.0
    for c in range(d.ncon):
        if d.contact[c].dist < 0:
            max_pen = max(max_pen, -d.contact[c].dist)
    check("no deep penetrations", max_pen < 0.01, f"max {max_pen * 1000:.2f} mm")

    # 8. actuator force ranges (from jnt actuatorfrcrange; motors themselves unlimited)
    frc = np.array([m.jnt_actfrcrange[j] for j in robot.joint_ids])
    check(
        "joint torque limits sane",
        bool((frc[:, 1] > 0).all() and (frc[:, 1] <= 400).all()),
        f"min {frc[:, 1].min():.0f} Nm, max {frc[:, 1].max():.0f} Nm",
    )
    lim_ok = np.allclose(frc[:, 1], manifest.effort_limits(), rtol=1e-3)
    check("torque limits match manifest", bool(lim_ok))

    # 9. hold rollout another 2 s with small perturbation
    # 0.15 m/s shove: within the ankle-strategy capture limit of a
    # non-stepping PD hold (0.3 m/s puts the capture point at the toe edge).
    d.qvel[0] += 0.15
    for i in range(int(2.0 / m.opt.timestep)):
        robot.apply_pd(target)
        robot.step()
        if renderer and i % 40 == 0:
            renderer.update_scene(d)
            frames.append(renderer.render())
    grav = robot.projected_gravity()
    check(
        "survives 0.15 m/s push while standing",
        bool(np.isfinite(d.qpos).all() and grav[2] < -0.9),
        f"gravity_z_in_base={grav[2]:.3f}, base_z={robot.base_pos()[2]:.3f}",
    )

    if renderer and frames:
        import imageio

        out = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        video = out / "a3_ultra_settle.mp4"
        imageio.mimsave(video, frames, fps=25)
        print(f"video saved: {video}")

    return finish()


def finish() -> int:
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 50}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
