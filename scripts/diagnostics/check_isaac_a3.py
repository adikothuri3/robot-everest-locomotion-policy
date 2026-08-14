"""A3 Ultra Isaac Lab validation — PASS/FAIL checks (headless).

IsaacSim/PhysX is the default training backend, so this is the pre-flight for a
cloud run: it imports the generated 29-DOF URDF with the SAME converter settings
Holosoma uses (`merge_fixed_joints` / `replace_cylinders_with_capsules` from
`RobotAssetConfig`), then checks the resulting articulation against the canonical
manifest AND against the preset's `body_names` — the list Holosoma asserts on at
startup. It PD-holds the default pose and writes results/consistency/isaac.json
for the cross-sim comparison.

Run inside the Isaac venv:
    .venv-isaac/Scripts/python.exe scripts/diagnostics/check_isaac_a3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.converters import UrdfConverterCfg  # noqa: E402

import yaml  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def _preset_body_names() -> list[str]:
    """BODY_NAMES literal from the Holosoma preset (parsed, so holosoma need not import)."""
    import ast

    src = (REPO / "src/everest_locomotion/holosoma_ext/a3_ultra_presets.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "BODY_NAMES":
            return ast.literal_eval(node.value)
    raise AssertionError("BODY_NAMES not found in a3_ultra_presets.py")


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    manifest = yaml.safe_load((REPO / "configs" / "robots" / "a3_ultra.yaml").read_text(encoding="utf-8"))
    control_joints = [j for j in manifest["joint_order"] if j not in manifest["control"]["frozen_joints"]]
    default_pose = {k: float(v) for k, v in manifest["default_pose"]["joint_pos"].items()}
    gains = manifest["pd_gains_assumed"]

    sim = SimulationContext(SimulationCfg(dt=0.001, device="cuda:0"))
    check("simulation context on GPU", sim.device == "cuda:0", f"device={sim.device}")

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)

    urdf = REPO / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.urdf"
    robot_cfg = ArticulationCfg(
        prim_path="/World/A3",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            fix_base=False,
            # must mirror a3_ultra_presets.RobotAssetConfig — Holosoma passes
            # collapse_fixed_joints / replace_cylinder_with_capsule straight
            # through to this converter.
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                target_type="none",
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(manifest["default_pose"]["base_height_m"]) + 0.005),
            joint_pos=default_pose,
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*hip.*": gains["hip"]["kp"],
                    ".*knee.*": gains["knee"]["kp"],
                    ".*ankle.*": gains["ankle"]["kp"],
                    "waist.*": gains["waist"]["kp"],
                    ".*shoulder.*": gains["shoulder"]["kp"],
                    ".*elbow.*": gains["elbow"]["kp"],
                    ".*wrist.*": gains["wrist"]["kp"],
                },
                damping={
                    ".*hip.*": gains["hip"]["kd"],
                    ".*knee.*": gains["knee"]["kd"],
                    ".*ankle.*": gains["ankle"]["kd"],
                    "waist.*": gains["waist"]["kd"],
                    ".*shoulder.*": gains["shoulder"]["kd"],
                    ".*elbow.*": gains["elbow"]["kd"],
                    ".*wrist.*": gains["wrist"]["kd"],
                },
            )
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    check("articulation loads", True, f"{len(robot.joint_names)} joints, {len(robot.body_names)} bodies")

    check(
        "29 actuated joints match manifest",
        sorted(robot.joint_names) == sorted(control_joints),
        f"{len(robot.joint_names)} joints",
    )

    # The Holosoma IsaacSim backend asserts num_bodies == len(body_names) and
    # resolves every name; a mismatch here is a crash on the cloud box.
    preset_bodies = _preset_body_names()
    got, want = set(robot.body_names), set(preset_bodies)
    check(
        "articulation bodies match preset body_names",
        got == want,
        f"{len(got)} bodies"
        + ("" if got == want else f"; missing={sorted(want - got)[:4]} extra={sorted(got - want)[:4]}"),
    )

    masses = robot.data.default_mass[0].cpu().numpy()
    total = float(masses.sum())
    check("total mass matches official", abs(total - 60.18) < 0.35, f"{total:.3f} kg")

    limits_attr = getattr(robot.data, "joint_pos_limits", None)
    if limits_attr is None:
        limits_attr = robot.data.joint_limits
    limits = limits_attr[0].cpu().numpy()
    joints_manifest = manifest["joints"]
    bad = []
    for i, name in enumerate(robot.joint_names):
        lo, hi = joints_manifest[name]["limits_rad"]
        if abs(limits[i][0] - lo) > 0.01 or abs(limits[i][1] - hi) > 0.01:
            bad.append(name)
    check("joint limits match manifest", not bad, f"mismatches: {bad[:4]}" if bad else "all 29 within 0.01 rad")

    # settle under implicit PD for 2 s
    default_pos = robot.data.default_joint_pos.clone()
    for i in range(2000):
        robot.set_joint_position_target(default_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    root = robot.data.root_state_w[0]
    z = float(root[2])
    quat = root[3:7].cpu().numpy()  # w,x,y,z
    # gravity alignment: project world -z into base frame via quat
    w, x, y, zq = quat
    g_z = -(1 - 2 * (x * x + y * y))
    finite = bool(torch.isfinite(robot.data.joint_pos).all())
    check("settle: no NaN", finite)
    check("settle: upright at height", bool(g_z < -0.9 and 0.85 < z < 1.2), f"z={z:.3f}, g_z={g_z:.3f}")

    # dump for cross-sim comparison
    out = {
        "simulator": "isaacsim-5.1/isaaclab-2.3",
        "asset": str(urdf.relative_to(REPO)),
        "total_mass": total,
        "gravity": [0.0, 0.0, -9.81],
        "n_actuated": len(robot.joint_names),
        "joints": {
            n: {"range": [float(limits[i][0]), float(limits[i][1])]}
            for i, n in enumerate(robot.joint_names)
        },
        "bodies": {n: {"mass": float(masses[i])} for i, n in enumerate(robot.body_names)},
    }
    out_dir = REPO / "results" / "consistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "isaac.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'isaac.json'}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 50}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
