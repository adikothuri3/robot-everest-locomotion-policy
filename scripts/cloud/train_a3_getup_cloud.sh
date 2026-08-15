#!/bin/bash
# =============================================================================
# Turnkey cloud training: AgiBot A3 Ultra GET-UP policy (Holosoma PPO, IsaacSim)
# =============================================================================
# Run this ON a fresh Linux GPU instance (Lambda Cloud: Ubuntu 22.04, NVIDIA
# driver preinstalled; 24 GB VRAM recommended: A10, L40S, RTX 6000 Ada, A100).
#
#   git clone <this-repo-url> everest && cd everest
#   bash scripts/cloud/train_a3_getup_cloud.sh
#
# Environment overrides:
#   EXPS="a3-ultra-rollover a3-ultra-getup"   (default: BOTH policies, in order)
#     Any subset/order works; -fast-sac variants exist for every task.
#   NUM_ENVS=4096   ITERATIONS=<n>   SEED=1
#   WANDB_API_KEY=...           (optional; enables logger:wandb)
#
# WHY TWO POLICIES: the literature never made one policy rise from every
# posture. HumanUP = prone->supine ROLLOVER policy (98.3% on hardware) +
# supine GET-UP policy (78.3%); HoST = per-posture policies, sides handled by
# the supine one. Run #1 (single policy, all postures) plateaued at 30% ~= the
# supine share of our pose bank. So: a3-ultra-rollover trains on prone starts
# (success = settled on the back, 1 s), a3-ultra-getup trains on supine+side
# starts (success = locomotion handoff pose, 2 s). At runtime a projected-
# gravity router chains rollover -> getup -> locomotion.
#
# What this trains (see docs/research/getup_recipes.md and the getup extension
# src/everest_locomotion/holosoma_ext/a3_ultra_getup.py):
#   HoST-style get-up adapted to Holosoma: fallen-pose starts from the
#   drop-and-settle bank (assets/a3_ultra/getup/), staged rewards
#   (upright -> rise -> locomotion default pose -> stand still), 350 N assist
#   force on the torso annealed to zero by held-stand success rate, smoothness
#   penalties ramped in by PenaltyCurriculum, velocity-blowup terminations,
#   NO contact termination (ground contact is the task).
#   Success = holding the locomotion handoff pose 2 s (manifest getup.terminal).
#
# Backend: IsaacSim ONLY for real runs — holosoma's nightly matrix validates
# [isaacgym, isaacsim]; MJWarp NaNs under untrained-policy flailing and get-up
# is maximal flailing. (Local MJWarp = smoke tests only.)
#
# Watch during training (TensorBoard: logs/everest-a3/<run>/; env log_dict
# tags are logged WITHOUT a prefix):
#   getup_success_rate     -> target >0.95 for rollover (settled-on-back gate),
#                             >0.9 for getup (manifest handoff-pose gate)
#   getup_assist_scale     -> must anneal 1.0 -> 0.0 (success without assist
#                             is the only success that counts)
#   penalty_scale          -> smoothness ramp (0.1 -> 1.0)
#   average_episode_length -> must start near ~500 (=10 s), NOT ~10 (if it
#                             pins low, envs are dying at spawn — abort)
# A policy is only DONE when: success_rate > 0.9 AND assist_scale == 0.0
# AND penalty_scale == 1.0.
# =============================================================================
set -euo pipefail

EXPS="${EXPS:-a3-ultra-rollover a3-ultra-getup}"
NUM_ENVS="${NUM_ENVS:-4096}"
# Iteration budgets differ per algo (FastSAC counts differently than PPO):
# only override the experiment's registered default if ITERATIONS is set.
ITERATIONS="${ITERATIONS:-}"
SEED="${SEED:-1}"
HOLOSOMA_COMMIT="6e146b0af5d7cd8a39b8bb2ed05b977cf70445d3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$HOME}"

echo "== [1/6] Sanity =="
command -v nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA driver"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
test -f "$REPO_DIR/assets/a3_ultra/holosoma/a3_ultra_29dof.urdf" || {
  echo "ERROR: generated A3 asset missing; regenerate with scripts/convert/make_holosoma_asset.py"; exit 1; }
test -f "$REPO_DIR/assets/a3_ultra/getup/fallen_poses_meta.json" || {
  echo "ERROR: fallen-pose bank missing (assets/a3_ultra/getup). It is committed to"
  echo "the repo; if absent regenerate with scripts/getup/generate_fallen_poses.py"
  exit 1
}
python3 - "$REPO_DIR" <<'EOF'
import json, sys, pathlib
meta = json.loads((pathlib.Path(sys.argv[1]) / "assets/a3_ultra/getup/fallen_poses_meta.json").read_text())
n = sum(meta["counts"].values())
assert n >= 1000, f"pose bank too small: {n}"
assert "hinge_joint_names_tree_order" in meta, "pose bank meta missing joint names — regenerate"
print(f"pose bank OK: {n} poses, categories {meta['counts']}")
EOF

echo "== [2/6] Clone holosoma @ ${HOLOSOMA_COMMIT:0:7} =="
if [[ ! -d "$WORK/holosoma" ]]; then
  git clone https://github.com/amazon-far/holosoma.git "$WORK/holosoma"
fi
git -C "$WORK/holosoma" fetch --all --quiet || true
git -C "$WORK/holosoma" checkout -q "$HOLOSOMA_COMMIT"
cd "$WORK/holosoma"

echo "== [3/6] IsaacSim environment (conda env hssim) =="
# setup_isaacsim.sh installs its OWN miniconda under $HOME/.holosoma_deps
# (scripts/source_common.sh) and creates `hssim` there; it is idempotent via
# $HOME/.holosoma_deps/.env_setup_finished_hssim. Do not bootstrap a second
# conda — source_isaacsim_setup.sh activates from holosoma's CONDA_ROOT
# regardless of what is on PATH. (Same as scripts/cloud/train_a3_cloud.sh.)
bash scripts/setup_isaacsim.sh
# shellcheck disable=SC1091
source scripts/source_isaacsim_setup.sh

echo "== [4/6] Verify stack + config resolution =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in torch"
print("torch", torch.__version__, "| cuda", torch.version.cuda)
EOF
export EVEREST_A3_ASSET_ROOT="$REPO_DIR/assets/a3_ultra/holosoma"
export EVEREST_A3_GETUP_POSES="$REPO_DIR/assets/a3_ultra/getup"
python - "$REPO_DIR" <<'EOF'
import sys
from holosoma.utils.config_registry import load_file_presets
load_file_presets([sys.argv[1] + "/src/everest_locomotion/holosoma_ext/a3_ultra_getup.py"])
import everest_getup
exp = everest_getup.a3_ultra_getup
assert exp.env_class == "everest_getup.A3UltraGetupManager"
assert exp.simulator.config.sim.max_episode_length_s == 10.0
print("getup experiment config resolves OK:", sorted(exp.reward.terms))
EOF

LOGGER_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(logger:wandb)
  echo "wandb enabled"
fi
ITER_ARGS=()
if [[ -n "$ITERATIONS" ]]; then
  ITER_ARGS=(--algo.config.num_learning_iterations "$ITERATIONS")
fi
for EXP in $EXPS; do
  echo "== [5/6] Train: $EXP  sim=isaacsim envs=$NUM_ENVS iters=${ITERATIONS:-exp-default} seed=$SEED =="
  python src/holosoma/holosoma/train_agent.py "exp:$EXP" "${LOGGER_ARGS[@]}" \
    --import-file "$REPO_DIR/src/everest_locomotion/holosoma_ext/a3_ultra_getup.py" \
    --training.num_envs "$NUM_ENVS" \
    --training.seed "$SEED" \
    "${ITER_ARGS[@]}" \
    --algo.config.save_interval 2000

  echo "== [6/6] Collect artifacts for $EXP =="
  RUN_DIR=$(ls -dt "$WORK"/holosoma/logs/everest-a3/*/ | head -1)
  OUT="$REPO_DIR/checkpoints/cloud_getup_$(basename "$RUN_DIR")"
  mkdir -p "$OUT"
  cp "$RUN_DIR"/model_*.pt "$RUN_DIR"/model_*.onnx "$RUN_DIR"/holosoma_config.yaml "$OUT"/ 2>/dev/null || true
  cp "$RUN_DIR"/events.out.tfevents.* "$OUT"/ 2>/dev/null || true
  echo "artifacts in: $OUT"
done
echo "retrieve with: scp -r <instance>:$OUT ./checkpoints/"
echo
echo "Next (locally): cross-physics gate in MuJoCo classic:"
echo "  python scripts/eval/stability_suite.py --policy onnx --onnx checkpoints/<run>/model_*.onnx"
echo "then the chained get-up -> locomotion handoff test (E14)."
