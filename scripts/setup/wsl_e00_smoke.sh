#!/bin/bash
# E00 smoke: upstream G1 FastSAC on MJWarp, tiny scale, verifies training executes.
set -e
cd ~/holosoma
source .venv/hsmujoco/bin/activate
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:mjwarp \
  --training.num_envs 512 \
  --training.seed 1 \
  --algo.config.num_learning_iterations 300 \
  2>&1 | tail -60
