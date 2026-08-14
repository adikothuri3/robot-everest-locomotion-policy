#!/bin/bash
# Train an A3 Ultra policy with Holosoma inside WSL2.
#
# Backend default is IsaacSim (PhysX) — the training-validated backend for this
# project (notes/decisions.md). MJWarp is smoke-only: it NaNs under untrained-
# policy flailing. Override with SIMULATOR=mjwarp for short local smoke runs.
#
# Caveat: IsaacSim is not supported inside WSL2 on consumer setups, so the
# isaacsim path here only works if holosoma's `hssim` conda env was actually
# built. Real IsaacSim runs go to the cloud: scripts/cloud/train_a3_cloud.sh.
#
# Usage (from Windows):
#   wsl -d Ubuntu-24.04 -- env bash scripts/train/train_a3_wsl.sh [fastsac|ppo|getup] [extra args...]
#   wsl -d Ubuntu-24.04 -- env SIMULATOR=mjwarp bash scripts/train/train_a3_wsl.sh fastsac --training.num_envs 256
# Usage (inside WSL):
#   bash scripts/train/train_a3_wsl.sh fastsac --training.num_envs 1024
set -euo pipefail

TASK="${1:-fastsac}"
shift || true

SIMULATOR="${SIMULATOR:-isaacsim}"
HOLOSOMA_DIR="${HOLOSOMA_DIR:-$HOME/holosoma}"
# Repo root, resolved from this script (works from any cwd, Windows or WSL side).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMPORT_FILE="$REPO_DIR/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py"

case "$TASK" in
  fastsac) EXP="a3-ultra-fast-sac" ;;
  ppo)     EXP="a3-ultra-ppo" ;;
  everest) EXP="a3-ultra-fast-sac-everest" ;;
  getup|getup-fast-sac)
    # The get-up extension imports a3_ultra_presets itself, so it is the only
    # --import-file needed. Real runs: scripts/cloud/train_a3_getup_cloud.sh.
    [[ "$TASK" == "getup" ]] && EXP="a3-ultra-getup" || EXP="a3-ultra-getup-fast-sac"
    IMPORT_FILE="$REPO_DIR/src/everest_locomotion/holosoma_ext/a3_ultra_getup.py"
    ;;
  *) echo "unknown task: $TASK (use fastsac|ppo|everest|getup|getup-fast-sac)"; exit 1 ;;
esac

# The presets resolve the asset directory and the fallen-pose bank from here.
export EVEREST_A3_ASSET_ROOT="$REPO_DIR/assets/a3_ultra/holosoma"
export EVEREST_A3_GETUP_POSES="$REPO_DIR/assets/a3_ultra/getup"

cd "$HOLOSOMA_DIR"

case "$SIMULATOR" in
  isaacsim)
    # holosoma's setup_isaacsim.sh drops its sentinel in $HOME/.holosoma_deps
    # (scripts/source_common.sh WORKSPACE_DIR), not in the repo.
    SENTINEL="${HOLOSOMA_WORKSPACE:-$HOME/.holosoma_deps}/.env_setup_finished_${CONDA_ENV_NAME:-hssim}"
    if [[ ! -f "$SENTINEL" ]]; then
      cat >&2 <<EOF
ERROR: no IsaacSim env ($SENTINEL missing).
IsaacSim is the default backend but is not supported in WSL2 on consumer setups.
  * real training  -> cloud: bash scripts/cloud/train_a3_cloud.sh   (SIMULATOR=isaacsim)
  * local smoke    -> SIMULATOR=mjwarp bash scripts/train/train_a3_wsl.sh $TASK ...
  * try anyway     -> bash $HOLOSOMA_DIR/scripts/setup_isaacsim.sh, then re-run
EOF
      exit 1
    fi
    # shellcheck disable=SC1091
    source scripts/source_isaacsim_setup.sh
    ;;
  mjwarp)
    echo "WARNING: MJWarp is smoke-only (NaNs under untrained flailing) — not for real runs." >&2
    # shellcheck disable=SC1091
    source .venv/hsmujoco/bin/activate
    ;;
  *) echo "unknown simulator: $SIMULATOR (use isaacsim|mjwarp)"; exit 1 ;;
esac

exec python src/holosoma/holosoma/train_agent.py "exp:$EXP" "simulator:$SIMULATOR" \
  --import-file "$IMPORT_FILE" \
  "$@"
