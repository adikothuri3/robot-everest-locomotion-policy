#!/bin/bash
# Probe holosoma's G1 mjwarp env directly: zero actions, inspect NaN sources.
set -e
cd ~/holosoma/src/holosoma
source ~/holosoma/.venv/hsmujoco/bin/activate
python - <<'EOF'
import contextlib
import dataclasses
import torch

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_values import experiment, simulator
from holosoma.train_agent import training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class

seeding(0)
base = experiment.g1_29dof_fast_sac
cfg = dataclasses.replace(
    base,
    simulator=simulator.mjwarp,
    training=dataclasses.replace(base.training, num_envs=8, headless=True, seed=0, torch_deterministic=False),
)
with contextlib.ExitStack() as stack:
    stack.enter_context(training_context(cfg))
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device="cuda:0")
    obs = env.reset_all()
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            print(f"reset obs[{k}]: shape={tuple(v.shape)} nan={torch.isnan(v).any().item()} inf={torch.isinf(v).any().item()}")
    tm = env.terrain_manager.terrain_term
    for name in ("_base_heights",):
        t = getattr(tm, name, None)
        if t is not None:
            print(f"{name}: min={t.min().item():.3f} max={t.max().item():.3f} nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()}")
    # emulate untrained FastSAC: uniform actions over the full scaled range
    rc = env.robot_config
    lo = torch.tensor(rc.dof_pos_lower_limit_list, device="cuda:0")
    hi = torch.tensor(rc.dof_pos_upper_limit_list, device="cuda:0")
    dflt = torch.zeros(29, device="cuda:0")
    for i, jn in enumerate(rc.dof_names):
        dflt[i] = rc.init_state.default_joint_angles.get(jn, 0.0)
    scaling = torch.maximum((lo - dflt).abs(), (hi - dflt).abs()) / rc.control.action_scale
    torch.manual_seed(0)
    total_dones = 0
    for i in range(100):
        actions = (torch.rand(8, 29, device="cuda:0") * 2 - 1) * scaling
        obs, rew, dones, infos = env.step({"actions": actions})
        total_dones += int(dones.sum())
        bad = torch.isnan(rew).any().item() or torch.isinf(rew).any().item()
        obs_bad = any(
            torch.isnan(v).any().item() for v in obs.values() if isinstance(v, torch.Tensor)
        )
        rs = env.simulator.robot_root_states
        av = rs[:, 10:13].abs().max().item()
        z = rs[:, 2]
        if i in (0, 1, 2, 5, 10, 20, 50, 99) or bad or obs_bad:
            print(
                f"step {i}: rew_nan={bad} obs_nan={obs_bad} rew_mean={rew.float().mean().item():.4f} "
                f"dones={int(dones.sum())} max|ang_vel|={av:.1f} z=[{z.min().item():.3f},{z.max().item():.3f}]"
            )
        if bad or obs_bad:
            break
    print(f"probe done; total dones over {i+1} steps: {total_dones}")
EOF
