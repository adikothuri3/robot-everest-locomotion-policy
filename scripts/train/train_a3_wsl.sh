#!/bin/bash
# Train A3 Ultra locomotion with Holosoma inside WSL2.
#
# Usage (from Windows):  wsl -d Ubuntu-24.04 -- bash scripts/train/train_a3_wsl.sh [fastsac|ppo] [extra args...]
# Usage (inside WSL):    bash scripts/train/train_a3_wsl.sh fastsac --training.num_envs 1024
set -e

ALGO="${1:-fastsac}"
shift || true

REPO_WIN=/mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy
IMPORT_FILE=$REPO_WIN/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py

case "$ALGO" in
  fastsac) EXP="a3-ultra-fast-sac" ;;
  ppo)     EXP="a3-ultra-ppo" ;;
  *) echo "unknown algo: $ALGO (use fastsac|ppo)"; exit 1 ;;
esac

cd ~/holosoma
source .venv/hsmujoco/bin/activate
exec python src/holosoma/holosoma/train_agent.py "exp:$EXP" simulator:mjwarp \
  --import-file "$IMPORT_FILE" \
  "$@"
