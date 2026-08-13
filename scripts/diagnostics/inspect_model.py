"""Dump structural information from a MuJoCo model (joints, actuators, masses, limits).

Usage:
    python scripts/diagnostics/inspect_model.py [path/to/model.xml]

Defaults to the official A3 Ultra T2.5 MJCF in third_party.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO / "third_party" / "A3-A3U-robot-model" / "a3_ultra_t2d5" / "mjcf" / "a3_ultra_t2d5.xml"
)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"model: {path}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"timestep={model.opt.timestep}")
    print(f"total mass={mujoco.mj_getTotalmass(model):.4f} kg")

    print("\n== JOINTS (model order) ==")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = mujoco.mjtJoint(model.jnt_type[j]).name
        rng = model.jnt_range[j]
        limited = bool(model.jnt_limited[j])
        frc = model.jnt_actfrcrange[j] if hasattr(model, "jnt_actfrcrange") else None
        dof = model.jnt_dofadr[j]
        damping = model.dof_damping[dof] if jtype != "mjJNT_FREE" else None
        friction = model.dof_frictionloss[dof] if jtype != "mjJNT_FREE" else None
        print(
            f"[{j:2d}] {name:32s} {jtype:12s} range=({rng[0]:+.4f},{rng[1]:+.4f}) "
            f"limited={limited} actfrc={frc} damping={damping} frictionloss={friction}"
        )

    print("\n== ACTUATORS (model order) ==")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jid = model.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        ctrl = model.actuator_ctrlrange[a]
        frc = model.actuator_forcerange[a]
        print(
            f"[{a:2d}] {name:32s} -> joint {jname:32s} "
            f"ctrlrange=({ctrl[0]:+.1f},{ctrl[1]:+.1f}) forcerange=({frc[0]:+.1f},{frc[1]:+.1f})"
        )

    print("\n== BODIES ==")
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"[{b:2d}] {name:32s} mass={model.body_mass[b]:8.4f} inertia={model.body_inertia[b]}")

    print("\n== SITES ==")
    for s in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)
        print(f"[{s:2d}] {name:32s} pos(world, qpos0)={data.site_xpos[s]}")

    print("\n== SENSORS ==")
    kinds: dict[str, int] = {}
    for s in range(model.nsensor):
        t = mujoco.mjtSensor(model.sensor_type[s]).name
        kinds[t] = kinds.get(t, 0) + 1
    for t, n in sorted(kinds.items()):
        print(f"  {t}: {n}")

    print("\n== KEYFRAMES ==")
    for k in range(model.nkey):
        print(f"key[{k}] qpos[:7]={model.key_qpos[k][:7]}")

    # COM at default pose
    print(f"\nCOM (qpos0) = {data.subtree_com[0]}")
    lowest = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != 0:
            z = data.geom_xpos[g][2] - model.geom_rbound[g]
            lowest = min(lowest, z)
    print(f"approx lowest geom point at qpos0 (base z=1.3): {lowest:.4f}")


if __name__ == "__main__":
    main()
