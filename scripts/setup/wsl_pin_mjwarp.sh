#!/bin/bash
# Install the mujoco_warp commit Holosoma pins in scripts/setup_mujoco.sh.
# The uv setup path installs mujoco-warp from PyPI unpinned, which pulled an
# incompatible 3.11.0 (physics diverges for BOTH G1 and A3). Upstream pin:
#   MUJOCO_WARP_COMMIT=ecaef88917a3c90cd238bf76681ca770f58033df
# NOTE: even this best-known combo (mujoco 3.11.0 + mjwarp ecaef88 + warp
# 1.15.0) eventually NaNs under untrained-policy flailing — MJWarp is not
# training-validated upstream (nightly matrix = isaacgym/isaacsim only).
# Use MJWarp for smoke runs; real training goes through simulator:isaacsim.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/holosoma
source .venv/hsmujoco/bin/activate
uv pip install 'git+https://github.com/google-deepmind/mujoco_warp.git@ecaef88917a3c90cd238bf76681ca770f58033df'
uv pip install 'warp-lang==1.15.0' 'mujoco==3.11.0'
python -c "import mujoco_warp, mujoco, warp; print('mjwarp OK; mujoco', mujoco.__version__, 'warp', warp.__version__)"
