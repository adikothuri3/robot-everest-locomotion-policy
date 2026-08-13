# Third-party dependencies

External repositories are cloned here but git-ignored; this file records exactly what
to clone and at which commit. `scripts/setup/clone_third_party.ps1` (Windows) and
`scripts/setup/clone_third_party.sh` (WSL2/Linux) automate it.

| Project | URL | Pinned commit | License | Purpose |
| --- | --- | --- | --- | --- |
| A3-A3U-robot-model | https://github.com/AgibotTech/A3-A3U-robot-model | `589f508ff357447c610a3f3004419035ddc8f153` | Mulan PSL v2 | Official A3 / A3 Ultra URDF + MJCF + meshes (source of truth for the robot) |
| holosoma | https://github.com/amazon-far/holosoma | pinned at clone time — record below when cloned | Apache-2.0 | Primary RL baseline (FastSAC/PPO, MJWarp/IsaacSim, DR, terrain, pushes). Lives in WSL2 at `~/holosoma`, not in this tree |

Reference-only projects (studied, NOT vendored):
| Project | URL | License | Why not vendored |
| --- | --- | --- | --- |
| IsaacLab | https://github.com/isaac-sim/IsaacLab | BSD-3 | Installed as pip package in its own venv (conservative baseline) |
| agibot_x1_train | https://github.com/AgibotTech/agibot_x1_train | **none** | No license ⇒ code cannot be reused; conventions documented in docs/research/ |
| humanoid-gym | https://github.com/roboterax/humanoid-gym | BSD-3 (setup.py) | Legacy Isaac Gym Preview 4, Linux-only; methodology reference only |
