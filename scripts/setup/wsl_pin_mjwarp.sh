#!/bin/bash
# Install the mujoco_warp commit Holosoma pins in scripts/setup_mujoco.sh.
# The uv setup path installs mujoco-warp from PyPI unpinned, which pulled an
# incompatible 3.11.0 (physics diverges for BOTH G1 and A3). Upstream pin:
#   MUJOCO_WARP_COMMIT=ecaef88917a3c90cd238bf76681ca770f58033df
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/holosoma
source .venv/hsmujoco/bin/activate
uv pip install 'git+https://github.com/google-deepmind/mujoco_warp.git@ecaef88917a3c90cd238bf76681ca770f58033df'
python -c "import mujoco_warp, mujoco, warp; print('mjwarp OK; mujoco', mujoco.__version__, 'warp', warp.__version__)"
