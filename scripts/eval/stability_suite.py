"""Stability benchmark for A3 Ultra policies (MuJoCo, seeded, repeatable).

Scenario families:
  quiet_stand   : 10 s undisturbed
  push_sweep    : impulses from 8 directions x magnitudes, at stance
  terrain_sweep : procedural_rough difficulty x slope grid (stand/track zero cmd)
  friction_sweep: low-friction ground (slip proxy)

Metrics per rollout: survival, fall time, slip distance/events, tilt, tracking
error, torque saturation, action jitter, joint-limit events (see
everest_locomotion.evaluation.rollout).

Results: results/stability/<run_name>.json (+ per-scenario rows). Policies:
  --policy stand           PD hold of default pose (pipeline baseline; a
                           non-stepping PD cannot absorb large pushes — its
                           numbers are the floor, not a target)
  --policy onnx --onnx P   exported Holosoma policy (obs layout must match)

Usage:
    python scripts/eval/stability_suite.py --policy stand --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from everest_locomotion.robots.manifest import load_manifest
from everest_locomotion.sim_adapters.mujoco_adapter import MujocoRobot
from everest_locomotion.sim_adapters.mujoco_scene import model_with_terrain
from everest_locomotion.terrains import TerrainSpec, procedural_rough
from everest_locomotion.policies import PDStandPolicy, OnnxPolicy
from everest_locomotion.evaluation.rollout import Push, run_rollout

# Diagnostic-stiff hold gains for the stand baseline (see check_mujoco_model.py)
HOLD_MIN = {"ankle": (800.0, 20.0), "hip": (400.0, 10.0), "knee": (400.0, 10.0), "waist": (400.0, 10.0)}


def stiffen(robot: MujocoRobot) -> None:
    for i, jname in enumerate(robot.manifest.joint_order):
        for key, (kp, kd) in HOLD_MIN.items():
            if key in jname:
                robot.kp[i] = max(robot.kp[i], kp)
                robot.kd[i] = max(robot.kd[i], kd)


def make_policy(args, manifest):
    ctrl_idx = [manifest.joint_order.index(j) for j in manifest.control_joints]
    default_ctrl = manifest.default_pose_vector()[ctrl_idx]
    if args.policy == "stand":
        return PDStandPolicy(default_ctrl)
    if args.policy == "onnx":
        layout = args.obs_layout.split(",")
        return OnnxPolicy(args.onnx, layout, default_ctrl, manifest.action_scale)
    raise ValueError(args.policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["stand", "onnx"], default="stand")
    p.add_argument("--onnx")
    p.add_argument(
        "--obs-layout",
        default="base_ang_vel,projected_gravity,command,joint_pos,joint_vel,prev_action",
    )
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="reduced grid for smoke testing")
    args = p.parse_args()

    manifest = load_manifest()
    run_name = args.name or f"{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}"
    rows: list[dict] = []

    def record(family: str, params: dict, result) -> None:
        row = {"family": family, **params, **result.__dict__}
        row.pop("extras", None)
        rows.append(row)
        status = "OK " if result.survived else "FALL"
        print(f"[{status}] {family} {params}")

    def fresh_robot(model=None) -> MujocoRobot:
        r = MujocoRobot(manifest, model=model)
        if args.policy == "stand":
            stiffen(r)
        return r

    # 1. quiet stand
    record("quiet_stand", {}, run_rollout(fresh_robot(), make_policy(args, manifest), 10.0))

    # 2. push sweep
    dirs = {"front": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1),
            "fl": (0.707, 0.707), "fr": (0.707, -0.707), "bl": (-0.707, 0.707), "br": (-0.707, -0.707)}
    mags = [0.1, 0.2] if args.quick else [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for dname, (dx, dy) in (list(dirs.items())[:4] if args.quick else dirs.items()):
        for mag in mags:
            res = run_rollout(
                fresh_robot(), make_policy(args, manifest), 6.0,
                pushes=[Push(time_s=2.0, velocity_xyz=(dx * mag, dy * mag, 0.0))],
            )
            record("push", {"direction": dname, "magnitude_ms": mag}, res)

    # 3. terrain sweep
    difficulties = [0.2, 0.6] if args.quick else [0.2, 0.4, 0.6, 0.8, 1.0]
    slopes = [0.0] if args.quick else [0.0, 5.0, 10.0]
    for diff in difficulties:
        for slope in slopes:
            spec = TerrainSpec(seed=args.seed, difficulty=diff, slope_deg=slope,
                               size_m=(10.0, 10.0), roughness_m=0.08)
            patch = procedural_rough(spec)
            model = model_with_terrain(manifest, patch)
            res = run_rollout(fresh_robot(model), make_policy(args, manifest), 6.0,
                              terrain_height_fn=patch.height_at)
            record("terrain", {"difficulty": diff, "slope_deg": slope}, res)

    # 4. friction sweep
    for mu in ([0.4] if args.quick else [0.7, 0.4, 0.25, 0.15]):
        robot = fresh_robot()
        robot.model.geom_friction[:, 0] = mu
        res = run_rollout(robot, make_policy(args, manifest), 6.0,
                          pushes=[Push(time_s=2.0, velocity_xyz=(0.1, 0.0, 0.0))])
        record("friction", {"mu": mu}, res)

    # summary
    survived = sum(r["survived"] for r in rows)
    push_rows = [r for r in rows if r["family"] == "push"]
    max_push = {}
    for r in push_rows:
        if r["survived"]:
            d = r["direction"]
            max_push[d] = max(max_push.get(d, 0.0), r["magnitude_ms"])
    summary = {
        "run_name": run_name,
        "policy": args.policy,
        "onnx": args.onnx,
        "seed": args.seed,
        "n_scenarios": len(rows),
        "n_survived": survived,
        "survival_rate": survived / len(rows),
        "max_recoverable_push_ms": max_push,
        "mean_slip_distance_m": float(np.mean([r["slip_distance_m"] for r in rows])),
        "mean_torque_saturation": float(np.mean([r["torque_saturation_frac"] for r in rows])),
    }

    out_dir = Path(__file__).resolve().parents[2] / "results" / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nsurvival {survived}/{len(rows)}  max_push={max_push}")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
