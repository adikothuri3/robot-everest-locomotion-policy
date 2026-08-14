"""Diagnostic: which get-up termination fires, and is it physical or NaN?

Steps the getup env directly with ZERO actions (pure PD pull toward default
pose from fallen starts) and attributes every would-be termination to its
cause. Run inside the holosoma env (WSL, mjwarp):

    python probe_terminations.py --num-envs 32 --steps 300
"""

from __future__ import annotations

import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--import-file", required=True)
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--steps", type=int, default=300)
args = p.parse_args()

from holosoma.utils.config_registry import load_file_presets

load_file_presets([args.import_file])

import dataclasses

import everest_getup
import torch
from holosoma.config_values import simulator
from holosoma.utils.helpers import get_class
from holosoma.config_types.env import get_tyro_env_config

cfg = everest_getup.a3_ultra_getup
mjwarp = simulator.mjwarp
mjwarp = dataclasses.replace(
    mjwarp,
    config=dataclasses.replace(
        mjwarp.config,
        sim=dataclasses.replace(mjwarp.config.sim, max_episode_length_s=10.0),
        mujoco_warp=dataclasses.replace(mjwarp.config.mujoco_warp, njmax_per_env=1024),
    ),
)
cfg = dataclasses.replace(
    cfg,
    simulator=mjwarp,
    training=dataclasses.replace(cfg.training, num_envs=args.num_envs, headless=True),
)

from holosoma.utils.eval_utils import init_sim_imports

init_sim_imports(cfg)
env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device="cuda:0")
env.reset_all()

zero = torch.zeros(env.num_envs, env.num_dof, device=env.device)
lim = env.getup_dof_vel_limits  # name-mapped, correct on every backend
use_random = bool(int(__import__("os").environ.get("PROBE_RANDOM", "0")))
print("actions:", "random sigma=0.8" if use_random else "zero")

for step in range(args.steps):
    act = torch.randn_like(zero) * 0.8 if use_random else zero
    env.step({"actions": act})
    dv = env.simulator.dof_vel[:, :].clone()
    rs = env.simulator.robot_root_states[:, 0:13].clone()
    ratio = (torch.abs(dv) / lim).amax()
    over3 = torch.any(torch.abs(dv) > 3.0 * lim, dim=1)
    base_v = torch.norm(rs[:, 7:10], dim=-1)
    nonfin = (~torch.isfinite(rs).all(dim=1)) | (~torch.isfinite(dv).all(dim=1))
    if step % 25 == 0 or nonfin.any():
        worst_j = (torch.abs(dv) / lim).amax(dim=0).argmax().item()
        print(
            f"step {step:4d}: resets={int(env.reset_buf.sum())} "
            f"max|v|/lim={ratio:.2f} (dof {env.dof_names[worst_j]}) "
            f"n_over8x={int(over3.sum())} max_base_v={base_v.max():.2f} "
            f"n_nonfinite={int(nonfin.sum())}",
            flush=True,
        )
    if nonfin.any():
        print(f"NONFINITE at step {step}: envs {nonfin.nonzero().flatten().tolist()[:8]}")
    if step == 0:
        print("lim min/max:", float(lim.min()), float(lim.max()))
    if over3.any():
        e = over3.nonzero().flatten()[0].item()
        j = (torch.abs(dv[e]) / lim).argmax().item()
        print(f"  OVER8 detail step {step}: env {e} dof {env.dof_names[j]} v={float(dv[e,j]):.1f} lim={float(lim[j]):.2f} epl={int(env.episode_length_buf[e])}")
print("probe done")
