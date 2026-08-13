#!/bin/bash
# A3 Ultra smoke training: FastSAC on MJWarp (GPU), short run with frequent
# logging; saves checkpoint + ONNX at the end. Validates the full A3 chain.
set -e
cd ~/holosoma
source .venv/hsmujoco/bin/activate
python -u src/holosoma/holosoma/train_agent.py exp:a3-ultra-fast-sac simulator:mjwarp \
  --import-file /mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py \
  --training.num_envs 512 \
  --training.seed 1 \
  --algo.config.num_learning_iterations 200 \
  --algo.config.logging_interval 10 \
  --algo.config.save_interval 200 \
  2>&1 | tail -80
