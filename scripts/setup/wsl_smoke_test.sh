#!/bin/bash
# Smoke test for the Holosoma MJWarp environment in WSL2.
set -e
cd ~/holosoma
source .venv/hsmujoco/bin/activate
python - <<'EOF'
import mujoco
print("mujoco", mujoco.__version__)
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
import warp as wp
wp.init()
print("warp devices:", [str(d) for d in wp.get_devices()])
import mujoco_warp
print("mujoco_warp OK")
import holosoma
print("holosoma import OK")
EOF
