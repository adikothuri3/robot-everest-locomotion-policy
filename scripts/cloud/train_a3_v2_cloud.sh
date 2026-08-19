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
# Stages (each is graded against v1 and promoted only if it wins). Iteration
# counts for s2+ are CUMULATIVE, because a resumed run continues `global_step`:
#   s0       every feature OFF                    ->  10k   ~35 min   (standalone)
#   s1       + heading, velocity estimator, history ->  50k   ~3.3 h   (from scratch)
#   s2       + arm curriculum, CAM               -> 130k   ~4.3 h    (continues s1)
#   s3       + slope terrain, gait quality       -> 230k   ~5.4 h    (continues s2)
#   s4       + smoothness reward terms           -> 260k   ~1.6 h    (continues s3)
#   s4-lcp   + Lipschitz penalty (ablation)      -> 260k   ~3-4 h (eager, fp32)
#   t1       + tracking precision, wide friction ->  90k   ~4.9 h    (COLD)
#
# t1 is NOT part of the s2->s4 arm/terrain ladder — it is the tracking-precision
# branch off S1 (docs/final_rl_policy.md is the ladder; the T1 rationale lives in
# notes/decisions.md 2026-08-18). It was meant to continue S1 in ~2.2 h, but only
# .onnx was pulled off the v2 runs and the instance is gone, so its 90k is a cold
# budget. Run it as:
#
#   bash scripts/cloud/train_a3_v2_cloud.sh t1
#
# and if an S1 .pt ever turns up, the same stage continues it instead:
#   RESUME_FROM=checkpoints/<s1-run>/model_0045000.pt #     bash scripts/cloud/train_a3_v2_cloud.sh t1
#
# S1..S4 SHARE ONE OBSERVATION CONTRACT (692 actor / 707 critic) precisely so
# that each stage can continue the previous one's weights. `FastSACAgent.load`
# restores actor, critic, optimizers, log_alpha, both normalizers and
# `global_step`, but has no padding path for a widened observation — so holding
# the vector fixed is what makes the ladder a curriculum instead of four
# unrelated runs. s0 is the odd one out (100 dims): it is the refactor-neutrality
# control against v1 and is never resumed from or into.
#
# Start mid-ladder with RESUME_FROM=<previous stage's model_*.pt>.
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
    s0|s1|s2|s3|s4|s4-lcp|t1) STAGES+=("$arg") ;;
    *) echo "unknown stage: $arg (use s0|s1|s2|s3|s4|s4-lcp|t1|all|rest)"; exit 1 ;;
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
# S1..S4 share one observation contract, so each stage CONTINUES the previous
# stage's weights (see the ladder comment in a3_ultra_loco_v2.py). PREV_CKPT is
# the .pt the next stage resumes from; RESUME_FROM seeds it when you start the
# queue mid-ladder, e.g.
#   RESUME_FROM=checkpoints/cloud_..._s1-locomotion/model_0050000.pt \
#     bash scripts/cloud/train_a3_v2_cloud.sh s2 s3 s4
PREV_CKPT="${RESUME_FROM:-}"

# Fail here, not four hours in. `FastSACAgent.load` needs the TRAINING checkpoint
# (.pt: actor, critic, both optimizers, log_alpha, both observation normalizers
# and global_step). A .onnx is an inference export and carries none of that, so
# passing one either errors on load or — worse — is ignored and the stage trains
# from scratch with a cumulative iteration count.
#
# Note that only .onnx files were ever pulled off the v2 instances: every
# checkpoints/cloud_2026081[5-7]_* directory here is ONNX-only. If you need to
# resume one of those stages, the .pt has to come off the Lambda box first.
if [[ -n "$PREV_CKPT" ]]; then
  case "$PREV_CKPT" in
    *.pt) ;;
    *) echo "ERROR: RESUME_FROM must be a training checkpoint (.pt), got: $PREV_CKPT"
       echo "       A .onnx is an inference export — it cannot seed a resumed run."
       exit 1 ;;
  esac
  if [[ ! -f "$PREV_CKPT" ]]; then
    echo "ERROR: RESUME_FROM does not exist: $PREV_CKPT"
    exit 1
  fi
fi

for STAGE in "${STAGES[@]}"; do
  EXP="a3-ultra-loco-v2-${STAGE}"
  echo
  echo "== [5/6] Train $EXP  sim=$SIMULATOR envs=$NUM_ENVS seed=$SEED code=$COMMIT_SHA =="
  date -u "+     started %Y-%m-%d %H:%M:%SZ"

  ITER_ARGS=()
  if [[ -n "${ITERATIONS:-}" ]]; then
    ITER_ARGS=(--algo.config.num-learning-iterations "$ITERATIONS")
  fi

  # s0 is the standalone neutrality control and s1 is the root of the chain;
  # neither ever resumes. s2+ resume from whatever the previous stage produced.
  RESUME_ARGS=()
  case "$STAGE" in
    s0|s1) ;;
    t1)
      # t1's 90k is a COLD budget, so it runs with or without a checkpoint. It
      # will happily continue an S1 `.pt` if one ever turns up (same 692/707
      # contract) — that just makes it finish sooner.
      if [[ -n "$PREV_CKPT" ]]; then
        RESUME_ARGS=(--training.checkpoint "$PREV_CKPT")
        echo "     continuing from: $PREV_CKPT"
      else
        echo "     training from scratch (90k, ~4.9 h) — no S1 .pt available"
      fi
      ;;
    *)
      if [[ -n "$PREV_CKPT" ]]; then
        RESUME_ARGS=(--training.checkpoint "$PREV_CKPT")
        echo "     resuming from: $PREV_CKPT"
      else
        echo "!! $EXP has no checkpoint to continue from — it would train from"
        echo "   scratch with a cumulative iteration count and waste the run."
        echo "   Pass RESUME_FROM=<path to the previous stage's model_*.pt>."
        SUMMARY+=("$STAGE  SKIPPED (no checkpoint to resume)")
        continue
      fi
      ;;
  esac

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
    "${RESUME_ARGS[@]}" \
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
  cp "$RUN_DIR"/model_*.onnx "$RUN_DIR"/holosoma_config.yaml "$OUT"/ 2>/dev/null || true
  cp "$RUN_DIR"/events.out.tfevents.* "$OUT"/ 2>/dev/null || true

  # The .pt files are the ONLY resumable artifact and they are what went missing
  # from every v2 run: the whole ladder was pulled down as ONNX, which cannot
  # seed a continuation, and by the time that was noticed the instance was gone.
  # Copy them loudly and count them — never `2>/dev/null || true`.
  N_PT=0
  if compgen -G "$RUN_DIR/model_*.pt" >/dev/null; then
    cp "$RUN_DIR"/model_*.pt "$OUT"/
    N_PT=$(find "$OUT" -maxdepth 1 -name 'model_*.pt' | wc -l)
  fi
  N_ONNX=$(find "$OUT" -maxdepth 1 -name 'model_*.onnx' | wc -l)
  echo "$COMMIT_SHA" > "$OUT/CODE_COMMIT"
  echo "artifacts in: $OUT  ($N_PT .pt, $N_ONNX .onnx, $(du -sh "$OUT" | cut -f1))"
  if [[ "$N_PT" -eq 0 ]]; then
    echo "!! NO .pt CHECKPOINTS COLLECTED for $EXP — this run cannot ever be resumed."
    echo "   Do not tear down this instance until you know why."
    SUMMARY+=("$STAGE  ok BUT NO .pt (unresumable) -> checkpoints/$(basename "$OUT")")
  else
    SUMMARY+=("$STAGE  ok ($N_PT .pt) -> checkpoints/$(basename "$OUT")")
  fi

  # hand the highest-iteration checkpoint to the next stage (sort -V so
  # model_0100000 beats model_0090000 instead of losing on string order)
  NEXT_CKPT=$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
  if [[ -n "$NEXT_CKPT" ]]; then
    PREV_CKPT="$NEXT_CKPT"
  else
    echo "!! no .pt produced by $EXP — the next stage will have nothing to resume"
    PREV_CKPT=""
  fi
done

echo
echo "=============================== QUEUE SUMMARY ==============================="
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo "============================================================================="

cat <<EOF

Retrieve locally -- PULL THE .pt FILES. They are the only resumable artifact,
they are ~20-30 MB each (a whole run is a few hundred MB), and every v2 run
before this one was pulled as ONNX only. That cost a 2 h continuation and forced
a 5 h cold retrain once the instance was gone.

  scp -r ubuntu@<instance>:$REPO_DIR/checkpoints/cloud_* ./checkpoints/

Then VERIFY before you tear the instance down -- this is the whole lesson:

  ls checkpoints/cloud_*/model_*.pt | wc -l     # must be > 0
  du -sh checkpoints/cloud_*/                   # ONNX-only dirs are suspiciously small

If that count is 0, the .pt files exist only on the box. Do not terminate it.

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
  ALL  Env/net_reward_per_step   must stay POSITIVE. If it goes negative, falling
                              early becomes optimal (there is no death penalty)
                              and episode length decays behind it. This is how
                              S2/S3/S4 burned 14 GPU-h while looking healthy.
  ALL  average_episode_length must climb toward the 1000 cap and stay there.
                              t1 abort rule: if it has not lifted off the floor by
                              ~5k iterations, the friction floor is too low for a
                              cold start — relaunch with a higher friction_lo
                              rather than letting it burn 5 hours.
  t1   Env/yaw_drift_dps      the number t1 exists to fix. Must fall.
  t1   Env/heading_err_deg    residual heading error over heading-mode envs.
  t1   Episode/rew_tracking_lin_vel_fine, _ang_vel_fine, rew_tracking_heading
                              the precision terms, each bounded [0, 1]. They
                              should RISE. Flat near 0 means the fine sigma is too
                              tight to give gradient; pinned near 1 means too loose.
  s1+  vel_est_rms_ms         must fall and stay low. A confidently wrong
                              estimate is worse than no estimate — this gates s1.
  s2+  Env/arm_amplitude      must WIDEN past 0.25. If it stalls, standing still
                              became the cheaper policy (docs/final_rl_policy.md §8).
  s2+  Env/cam_z, Env/cam_xy  sanity-check the CAM scale before trusting
                              rew_cam_tracking; its sigma and reference amplitude
                              are ASSUMED, not derived.
  all  average_episode_length drives both penalty and arm curricula.

Log every run as a row in notes/experiments.md with commit $COMMIT_SHA.
EOF
