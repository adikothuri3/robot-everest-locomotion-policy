"""Dump simulator-independent robot properties to JSON for cross-sim comparison.

MuJoCo side runs everywhere:
    python scripts/diagnostics/dump_model_properties.py --sim mujoco

Isaac side must run inside the Isaac venv (imports isaaclab):
    .venv-isaac/Scripts/python.exe scripts/diagnostics/dump_model_properties.py --sim isaac

Then compare:
    python scripts/diagnostics/compare_sim_properties.py results/consistency/mujoco.json results/consistency/isaac.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = REPO / "results" / "consistency"


def dump_mujoco() -> dict:
    import mujoco
    import numpy as np

    from everest_locomotion.robots.manifest import load_manifest
    from everest_locomotion.sim_adapters.mujoco_adapter import MujocoRobot

    manifest = load_manifest()
    xml = REPO / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    robot = MujocoRobot(manifest, model=model) if False else None  # adapter expects 31 dof; use raw model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joints = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        joints[name] = {
            "range": [float(x) for x in model.jnt_range[j]],
            "effort_limit": float(model.jnt_actfrcrange[j][1]),
            "damping": float(model.dof_damping[model.jnt_dofadr[j]]),
            "frictionloss": float(model.dof_frictionloss[model.jnt_dofadr[j]]),
        }
    bodies = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        bodies[name] = {
            "mass": float(model.body_mass[b]),
            "inertia_diag": [float(x) for x in model.body_inertia[b]],
        }
    return {
        "simulator": f"mujoco-{mujoco.__version__}",
        "asset": str(xml.relative_to(REPO)),
        "total_mass": float(mujoco.mj_getTotalmass(model)),
        "gravity": [float(x) for x in model.opt.gravity],
        "timestep": float(model.opt.timestep),
        "n_actuated": int(model.nu),
        "joints": joints,
        "bodies": bodies,
    }


def dump_isaac() -> dict:
    """Load the A3 URDF via Isaac Lab and dump the same properties."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app

    import torch
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext, SimulationCfg, UrdfFileCfg
    import isaaclab.sim as sim_utils

    sim = SimulationContext(SimulationCfg(dt=0.001))
    urdf = REPO / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.urdf"
    cfg = ArticulationCfg(
        prim_path="/World/A3",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.07)),
        actuators={},
    )
    robot = Articulation(cfg)
    sim.reset()

    names = robot.joint_names
    limits = robot.data.joint_pos_limits[0].cpu().numpy()
    masses = robot.data.default_mass[0].cpu().numpy()
    joints = {
        n: {"range": [float(limits[i][0]), float(limits[i][1])]} for i, n in enumerate(names)
    }
    bodies = {n: {"mass": float(masses[i])} for i, n in enumerate(robot.body_names)}
    out = {
        "simulator": "isaacsim-5.1",
        "asset": str(urdf.relative_to(REPO)),
        "total_mass": float(masses.sum()),
        "gravity": [0.0, 0.0, -9.81],
        "n_actuated": len(names),
        "joints": joints,
        "bodies": bodies,
    }
    app.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sim", choices=["mujoco", "isaac"], required=True)
    args = p.parse_args()
    data = dump_mujoco() if args.sim == "mujoco" else dump_isaac()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.sim}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"wrote {out} (total mass {data['total_mass']:.4f} kg, {data['n_actuated']} actuated)")


if __name__ == "__main__":
    main()
