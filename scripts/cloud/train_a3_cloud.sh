#!/bin/bash
# =============================================================================
# Turnkey cloud training: AgiBot A3 Ultra stable walking (Holosoma FastSAC/MJWarp)
# =============================================================================
# Run this ON a fresh Linux GPU instance (Ubuntu 22.04 or 24.04, NVIDIA driver
# >= 555.58.02, 24 GB VRAM recommended: RTX 4090 / L40S / A10G-24 / A100).
#
#   git clone <this-repo-url> everest && cd everest
#   bash scripts/cloud/train_a3_cloud.sh                    # full stable-walk run
#
# Environment overrides:
#   EXP=a3-ultra-fast-sac|a3-ultra-fast-sac-everest|a3-ultra-ppo  (default: a3-ultra-fast-sac)
#   NUM_ENVS=4096      ITERATIONS=50000     SEED=1
#   WANDB_API_KEY=...  (optional; enables logger:wandb + auto ONNX upload)
#
# Everything this script pins/patches exists because it broke in bring-up:
#   * holosoma @ 6e146b0 (2026-08-11) — validated commit
#   * mujoco_warp @ ecaef88 — holosoma's own pin; the PyPI 3.11.0 release
#     produces DIVERGING PHYSICS (NaN/1e12 velocities) with this holosoma
#     commit on BOTH G1 and A3 (verified 2026-08-13)
#   * warp_utils patch — Warp >= 1.16 removed wp.types.array
#   * A3 asset — generated variant (primitive collisions: official mesh
#     collisions overflow MJWarp constraint budget; armature; solref 0.01)
#
# Why this recipe for "immense stability": exp:a3-ultra-fast-sac is the
# upstream FastSAC sim-to-real recipe unchanged except for the robot — it
# already trains WITH rough-terrain mix, periodic push perturbations, friction/
# mass/CoM/PD-gain/torque-RFI/latency randomization, and an action-rate
# curriculum, and is hardware-validated on two humanoids. Train this first;
# then compare exp:a3-ultra-fast-sac-everest (adds stumble/foothold/joint-limit
# penalties + stronger alive weight) on the stability suite and keep the winner.
# =============================================================================
set -euo pipefail

EXP="${EXP:-a3-ultra-fast-sac}"
NUM_ENVS="${NUM_ENVS:-4096}"
ITERATIONS="${ITERATIONS:-50000}"
SEED="${SEED:-1}"
HOLOSOMA_COMMIT="6e146b0af5d7cd8a39b8bb2ed05b977cf70445d3"
MUJOCO_WARP_COMMIT="ecaef88917a3c90cd238bf76681ca770f58033df"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$HOME}"

echo "== [1/6] Sanity =="
command -v nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA driver"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
test -f "$REPO_DIR/assets/a3_ultra/holosoma/a3_ultra_29dof.xml" || {
  echo "ERROR: generated A3 asset missing. Run on a machine with the asset committed,"
  echo "or regenerate: python scripts/convert/make_holosoma_asset.py (needs third_party model)"
  exit 1
}

echo "== [2/6] Clone holosoma @ ${HOLOSOMA_COMMIT:0:7} =="
if [[ ! -d "$WORK/holosoma" ]]; then
  git clone https://github.com/amazon-far/holosoma.git "$WORK/holosoma"
fi
git -C "$WORK/holosoma" fetch --all --quiet || true
git -C "$WORK/holosoma" checkout -q "$HOLOSOMA_COMMIT"

echo "== [3/6] Environment (uv, Python auto: 22.04->3.10 / 24.04->3.12) =="
cd "$WORK/holosoma"
bash scripts/setup_mujoco_via_uv.sh --no-robot-sdks
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source .venv/hsmujoco/bin/activate

echo "== [4/6] Pin mujoco_warp + patch warp_utils =="
uv pip install "git+https://github.com/google-deepmind/mujoco_warp.git@${MUJOCO_WARP_COMMIT}"
python "$REPO_DIR/scripts/setup/patch_holosoma_warp.py" \
  "$WORK/holosoma/src/holosoma/holosoma/utils/warp_utils.py"
python - <<'EOF'
import mujoco, warp, mujoco_warp, torch
assert torch.cuda.is_available(), "CUDA not available in torch"
print("stack OK:", "mujoco", mujoco.__version__, "| warp", warp.__version__, "| cuda", torch.version.cuda)
EOF

echo "== [5/6] Train: $EXP  envs=$NUM_ENVS iters=$ITERATIONS seed=$SEED =="
export EVEREST_A3_ASSET_ROOT="$REPO_DIR/assets/a3_ultra/holosoma"
LOGGER_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(logger:wandb)
  echo "wandb enabled"
fi
python src/holosoma/holosoma/train_agent.py "exp:$EXP" simulator:mjwarp "${LOGGER_ARGS[@]}" \
  --import-file "$REPO_DIR/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py" \
  --training.num_envs "$NUM_ENVS" \
  --training.seed "$SEED" \
  --algo.config.num_learning_iterations "$ITERATIONS" \
  --algo.config.save_interval 5000

echo "== [6/6] Collect artifacts =="
RUN_DIR=$(ls -dt "$WORK"/holosoma/logs/everest-a3/*/ | head -1)
OUT="$REPO_DIR/checkpoints/cloud_$(basename "$RUN_DIR")"
mkdir -p "$OUT"
cp "$RUN_DIR"/model_*.pt "$RUN_DIR"/model_*.onnx "$RUN_DIR"/holosoma_config.yaml "$OUT"/ 2>/dev/null || true
cp -r "$RUN_DIR"/events.out.tfevents.* "$OUT"/ 2>/dev/null || true
echo "artifacts in: $OUT"
echo "retrieve with: scp -r <instance>:$OUT ./checkpoints/"
echo
echo "Next (locally): validate cross-physics + stability:"
echo "  python scripts/eval/stability_suite.py --policy onnx --onnx checkpoints/<run>/model_*.onnx"
