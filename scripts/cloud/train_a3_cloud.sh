#!/bin/bash
# =============================================================================
# Turnkey cloud training: AgiBot A3 Ultra stable walking (Holosoma FastSAC)
# =============================================================================
# Run this ON a fresh Linux GPU instance (Ubuntu 22.04 or 24.04, NVIDIA driver
# >= 555.58.02, 24 GB VRAM recommended: RTX 4090 / L40S / A10G / L4 / A100).
#
#   git clone <this-repo-url> everest && cd everest
#   bash scripts/cloud/train_a3_cloud.sh                    # full stable-walk run
#
# Environment overrides:
#   SIMULATOR=isaacsim|mjwarp   (default isaacsim — see WHY below)
#   EXP=a3-ultra-fast-sac|a3-ultra-fast-sac-everest|a3-ultra-ppo
#   NUM_ENVS=4096   ITERATIONS=50000   SEED=1
#   WANDB_API_KEY=...           (optional; enables logger:wandb + ONNX upload)
#
# WHY isaacsim: holosoma's own nightly training matrix validates FastSAC on
# [isaacgym, isaacsim] ONLY. The MJWarp backend is smoke-tested but NOT
# training-validated upstream, and we reproduced deterministic physics NaNs
# under early-training flailing on MJWarp (upstream G1 asset, pinned versions,
# 2026-08-13) — the A3 port was not the cause (control experiment). Use MJWarp
# only for short local smoke runs.
#
# Pins/patches (each one broke during bring-up — do not unpin casually):
#   * holosoma @ 6e146b0 (2026-08-11) — validated commit
#   * (mjwarp path only) mujoco_warp @ ecaef88 = holosoma's own pin; PyPI
#     mujoco-warp 3.11.0 diverges violently. Best local combo found:
#     mujoco 3.11.0 + mujoco_warp ecaef88 + warp-lang 1.15.0 — still NaNs
#     under sustained random flailing, hence isaacsim default.
#   * (mjwarp path only) warp_utils patch — Warp >= 1.16 removed wp.types.array
#   * A3 asset — generated (primitive collisions, armature, solref 0.01)
#
# Recipe for "immense stability": exp:a3-ultra-fast-sac is the upstream FastSAC
# sim-to-real recipe unchanged except for the robot — rough-terrain mix, pushes
# every 5-10 s, friction/mass/CoM/PD/torque-RFI/latency randomization, action-
# rate curriculum; hardware-validated on two humanoids. Train it first, then
# exp:a3-ultra-fast-sac-everest (stumble/foothold/joint-limit penalties,
# alive 15) and keep the stability-suite winner.
# =============================================================================
set -euo pipefail

SIMULATOR="${SIMULATOR:-isaacsim}"
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
  echo "ERROR: generated A3 asset missing (assets/a3_ultra/holosoma). It is committed"
  echo "to the repo; if absent regenerate with scripts/convert/make_holosoma_asset.py"
  exit 1
}

echo "== [2/6] Clone holosoma @ ${HOLOSOMA_COMMIT:0:7} =="
if [[ ! -d "$WORK/holosoma" ]]; then
  git clone https://github.com/amazon-far/holosoma.git "$WORK/holosoma"
fi
git -C "$WORK/holosoma" fetch --all --quiet || true
git -C "$WORK/holosoma" checkout -q "$HOLOSOMA_COMMIT"
cd "$WORK/holosoma"

if [[ "$SIMULATOR" == "isaacsim" ]]; then
  echo "== [3/6] IsaacSim environment (conda env hssim; Ubuntu 22.04+) =="
  # setup_isaacsim.sh installs its OWN miniconda under $HOME/.holosoma_deps
  # (scripts/source_common.sh) and creates the `hssim` env there; it is
  # idempotent via $HOME/.holosoma_deps/.env_setup_finished_hssim. Do not
  # bootstrap a second conda — source_isaacsim_setup.sh activates from
  # holosoma's own CONDA_ROOT regardless of what is on PATH.
  bash scripts/setup_isaacsim.sh
  # shellcheck disable=SC1091
  source scripts/source_isaacsim_setup.sh
  echo "== [4/6] Verify stack =="
  python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in torch"
print("torch", torch.__version__, "| cuda", torch.version.cuda)
EOF
else
  echo "== [3/6] MJWarp environment (uv; local-smoke quality only) =="
  bash scripts/setup_mujoco_via_uv.sh --no-robot-sdks
  export PATH="$HOME/.local/bin:$PATH"
  # shellcheck disable=SC1091
  source .venv/hsmujoco/bin/activate
  echo "== [4/6] Pin known-best mjwarp combo + patch warp_utils =="
  uv pip install "git+https://github.com/google-deepmind/mujoco_warp.git@${MUJOCO_WARP_COMMIT}"
  uv pip install "warp-lang==1.15.0" "mujoco==3.11.0"
  python "$REPO_DIR/scripts/setup/patch_holosoma_warp.py" \
    "$WORK/holosoma/src/holosoma/holosoma/utils/warp_utils.py"
fi

echo "== [5/6] Train: $EXP  sim=$SIMULATOR envs=$NUM_ENVS iters=$ITERATIONS seed=$SEED =="
export EVEREST_A3_ASSET_ROOT="$REPO_DIR/assets/a3_ultra/holosoma"
LOGGER_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(logger:wandb)
  echo "wandb enabled"
fi
python src/holosoma/holosoma/train_agent.py "exp:$EXP" "simulator:$SIMULATOR" "${LOGGER_ARGS[@]}" \
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
cp "$RUN_DIR"/events.out.tfevents.* "$OUT"/ 2>/dev/null || true
echo "artifacts in: $OUT"
echo "retrieve with: scp -r <instance>:$OUT ./checkpoints/"
echo
echo "Next (locally): cross-physics + stability validation:"
echo "  python scripts/eval/stability_suite.py --policy onnx --onnx checkpoints/<run>/model_*.onnx"
