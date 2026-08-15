#!/bin/bash
# =============================================================================
# Turnkey cloud training: A3 Ultra **final walking policy v2** (Holosoma FastSAC)
# =============================================================================
# Runs the staged ladder from docs/final_rl_policy.md on a fresh Linux GPU box
# (Lambda Cloud, Ubuntu 22.04/24.04, NVIDIA driver >= 555.58.02, >= 24 GB VRAM:
# A100 / H100 / L40S / A10 / RTX 6000 Ada).
#
#   git clone <this-repo-url> everest && cd everest
#   bash scripts/cloud/train_a3_v2_cloud.sh s0            # 10k it, ~35 min
#   bash scripts/cloud/train_a3_v2_cloud.sh s1            # 50k it, ~2.7 h
#   bash scripts/cloud/train_a3_v2_cloud.sh all           # s0 -> s4, ~16-19 GPU-h
#   bash scripts/cloud/train_a3_v2_cloud.sh rest          # s1 -> s4 (s0 already done)
#   bash scripts/cloud/train_a3_v2_cloud.sh s1 s3 s4      # any queue you like
#
# Stages run back-to-back unattended. A stage that FAILS does not abort the queue
# (a transient death at hour 6 would otherwise cost every remaining stage); the
# run prints a QUEUE SUMMARY at the end saying which stages produced artifacts.
#
# Stages (each is graded against v1 and promoted only if it wins):
#   s0       every feature OFF                      10k   ~35 min
#   s1       + heading, velocity estimator, history 50k   ~2.7 h
#   s2       + arm curriculum, CAM                  80k   ~4.3 h
#   s3       + height scan, slope terrain          100k   ~5.4 h+
#   s4       + smoothness reward terms              30k   ~1.6 h
#   s4-lcp   + Lipschitz penalty (ablation)         30k   ~3-4 h (eager, fp32)
#
# STAGES ARE NOT RESUMED FROM EACH OTHER. Every stage changes the observation
# vector (s0 100 dims -> s1 575 -> s2 645 -> s3 692), so no checkpoint transfers.
# Each stage trains from scratch; that is what the cost model already assumes.
#
# Environment overrides:
#   NUM_ENVS=4096   SEED=1   SIMULATOR=isaacsim
#   ITERATIONS=...              override the stage's iteration count
#   WANDB_API_KEY=...           optional; enables logger:wandb + ONNX upload
#   EXTRA_ARGS="--foo bar"      passed through to train_agent.py
#
# Pins (each one broke during bring-up — do not unpin casually):
#   * holosoma @ 6e146b0 (2026-08-11) — validated commit
#   * A3 asset — generated, committed (scripts/convert/make_holosoma_asset.py)
#
# NOTE: no Holosoma fork is needed. The velocity estimator and the Lipschitz
# ablation ship as a FastSACAgent subclass inside the --import-file extension,
# reached through `algo._target_` (see notes/decisions.md 2026-08-15).
# =============================================================================
set -euo pipefail

SIMULATOR="${SIMULATOR:-isaacsim}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-1}"
HOLOSOMA_COMMIT="6e146b0af5d7cd8a39b8bb2ed05b977cf70445d3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$HOME}"
EXT="$REPO_DIR/src/everest_locomotion/holosoma_ext/a3_ultra_loco_v2.py"

# Accepts any number of stages, run back-to-back unattended:
#   ... s1                 one stage
#   ... s1 s2 s3 s4        a queue (skip s0 if it already ran)
#   ... all                s0 s1 s2 s3 s4
#   ... rest               s1 s2 s3 s4
STAGES=()
for arg in "${@:-s1}"; do
  case "$arg" in
    all)  STAGES+=(s0 s1 s2 s3 s4) ;;
    rest) STAGES+=(s1 s2 s3 s4) ;;
    s0|s1|s2|s3|s4|s4-lcp) STAGES+=("$arg") ;;
    *) echo "unknown stage: $arg (use s0|s1|s2|s3|s4|s4-lcp|all|rest)"; exit 1 ;;
  esac
done

echo "== [1/6] Sanity =="
command -v nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA driver"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
test -f "$REPO_DIR/assets/a3_ultra/holosoma/a3_ultra_29dof.xml" || {
  echo "ERROR: generated A3 asset missing (assets/a3_ultra/holosoma). It is committed"
  echo "to the repo; if absent regenerate with scripts/convert/make_holosoma_asset.py"
  exit 1
}
test -f "$EXT" || { echo "ERROR: extension missing: $EXT"; exit 1; }

echo "== [2/6] Clone holosoma @ ${HOLOSOMA_COMMIT:0:7} =="
if [[ ! -d "$WORK/holosoma" ]]; then
  git clone https://github.com/amazon-far/holosoma.git "$WORK/holosoma"
fi
git -C "$WORK/holosoma" fetch --all --quiet || true
git -C "$WORK/holosoma" checkout -q "$HOLOSOMA_COMMIT"
cd "$WORK/holosoma"

echo "== [3/6] IsaacSim environment (conda env hssim) =="
# setup_isaacsim.sh installs its OWN miniconda under $HOME/.holosoma_deps and is
# idempotent via $HOME/.holosoma_deps/.env_setup_finished_hssim. Do not bootstrap
# a second conda — source_isaacsim_setup.sh activates from holosoma's CONDA_ROOT.
bash scripts/setup_isaacsim.sh
# shellcheck disable=SC1091
source scripts/source_isaacsim_setup.sh

echo "== [4/6] Verify stack =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in torch"
print("torch", torch.__version__, "| cuda", torch.version.cuda)
EOF

export EVEREST_A3_ASSET_ROOT="$REPO_DIR/assets/a3_ultra/holosoma"
LOGGER_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(logger:wandb)
  echo "wandb enabled"
fi

COMMIT_SHA="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo
echo "== queue: ${STAGES[*]} =="

SUMMARY=()
for STAGE in "${STAGES[@]}"; do
  EXP="a3-ultra-loco-v2-${STAGE}"
  echo
  echo "== [5/6] Train $EXP  sim=$SIMULATOR envs=$NUM_ENVS seed=$SEED code=$COMMIT_SHA =="
  date -u "+     started %Y-%m-%d %H:%M:%SZ"

  ITER_ARGS=()
  if [[ -n "${ITERATIONS:-}" ]]; then
    ITER_ARGS=(--algo.config.num-learning-iterations "$ITERATIONS")
  fi

  # A stage that dies must not take the rest of an overnight queue with it: a
  # transient failure at hour 6 would otherwise cost every remaining stage. Record
  # it, keep going, and report at the end.
  set +e
  # shellcheck disable=SC2086
  python src/holosoma/holosoma/train_agent.py "exp:$EXP" "simulator:$SIMULATOR" \
    "${LOGGER_ARGS[@]}" \
    --import-file "$EXT" \
    --training.num-envs "$NUM_ENVS" \
    --training.seed "$SEED" \
    "${ITER_ARGS[@]}" \
    ${EXTRA_ARGS:-}
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "!! $EXP FAILED (exit $RC) — continuing with the rest of the queue"
    SUMMARY+=("$STAGE  FAILED (exit $RC)")
    continue
  fi

  echo "== [6/6] Collect artifacts for $EXP =="
  RUN_DIR=$(ls -dt "$WORK"/holosoma/logs/everest-a3/*"${EXP//-/_}"-locomotion/ 2>/dev/null | head -1)
  if [[ -z "$RUN_DIR" ]]; then
    echo "!! no log dir found for $EXP"
    SUMMARY+=("$STAGE  trained but NO ARTIFACTS")
    continue
  fi
  OUT="$REPO_DIR/checkpoints/cloud_$(basename "$RUN_DIR")"
  mkdir -p "$OUT"
  cp "$RUN_DIR"/model_*.pt "$RUN_DIR"/model_*.onnx "$RUN_DIR"/holosoma_config.yaml "$OUT"/ 2>/dev/null || true
  cp "$RUN_DIR"/events.out.tfevents.* "$OUT"/ 2>/dev/null || true
  echo "$COMMIT_SHA" > "$OUT/CODE_COMMIT"
  echo "artifacts in: $OUT"
  SUMMARY+=("$STAGE  ok -> checkpoints/$(basename "$OUT")")
done

echo
echo "=============================== QUEUE SUMMARY ==============================="
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo "============================================================================="

cat <<EOF

Retrieve locally:
  scp -r ubuntu@<instance>:$REPO_DIR/checkpoints/cloud_* ./checkpoints/

Then grade (docs/final_rl_policy.md §7) — the harness reads each policy's
observation layout out of its own ONNX metadata, so no harness flags change
between stages:
  P=checkpoints/<run>/model_XXXXXXX.onnx
  python scripts/eval/sim2sim_suite.py --mode sweep     --run-dir checkpoints/<run>
  python scripts/eval/sim2sim_suite.py --mode grid      --onnx \$P --name v2-<stage> --with-baseline
  python scripts/eval/sim2sim_suite.py --mode showcase  --onnx \$P --name v2-<stage> --video
  python scripts/eval/sim2sim_suite.py --mode pushlimit --onnx \$P --name v2-<stage>
  python scripts/eval/sim2sim_arms.py  --onnx \$P                     # arms driven by a skill
  python scripts/eval/sim2sim_arms.py  --onnx \$P --mask-arm-obs      # ablation

Watch these scalars in TensorBoard/W&B while a stage runs:
  s1+  vel_est_rms_ms         must fall and stay low. A confidently wrong
                              estimate is worse than no estimate — this gates s1.
  s1+  Episode/rew_tracking_ang_vel   >= 0.8 (v1 finished at 0.559)
  s2+  Env/arm_amplitude      must WIDEN past 0.25. If it stalls, standing still
                              became the cheaper policy (docs/final_rl_policy.md §8).
  s2+  Env/cam_z, Env/cam_xy  sanity-check the CAM scale before trusting
                              rew_cam_tracking; its sigma and reference amplitude
                              are ASSUMED, not derived.
  all  average_episode_length drives both penalty and arm curricula.

Log every run as a row in notes/experiments.md with commit $COMMIT_SHA.
EOF
