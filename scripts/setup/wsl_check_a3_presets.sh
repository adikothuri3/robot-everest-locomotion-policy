#!/bin/bash
# Validate the A3 Ultra Holosoma presets: registry import + asset load via MuJoCo.
set -e
cd ~/holosoma
source .venv/hsmujoco/bin/activate
python - <<'EOF'
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "a3_presets",
    "/mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from holosoma.config_values.robot import ROBOT_REGISTRY
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
r = ROBOT_REGISTRY.get("a3_ultra_29dof")
print("robot registered:", r.asset.robot_type, "| dofs:", len(r.dof_names), "| bodies:", r.num_bodies)
for name in ("a3_ultra_fast_sac", "a3_ultra_ppo", "a3_ultra_fast_sac_everest"):
    e = EXPERIMENT_REGISTRY.get(name)
    print("experiment registered:", name, "| algo:", type(e.algo).__name__, "| sim:", e.simulator.config.name)

import mujoco
xml = os.path.join(r.asset.asset_root, r.asset.xml_file)
m = mujoco.MjModel.from_xml_path(xml)
print("asset loads in WSL:", xml, "| nu =", m.nu, "| mass =", round(mujoco.mj_getTotalmass(m), 3))

# cross-check dof names exist in model
missing = [d for d in r.dof_names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, d) < 0]
assert not missing, f"missing joints: {missing}"
bodies_in_model = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)]
assert r.body_names == bodies_in_model, "body_names order mismatch"
print("dof/body name cross-check OK")
EOF
