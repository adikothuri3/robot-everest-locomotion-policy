# Third-party dependencies

External repositories are cloned here but git-ignored; this file records exactly what
to clone and at which commit. `scripts/setup/clone_third_party.ps1` (Windows) and
`scripts/setup/clone_third_party.sh` (WSL2/Linux) automate it.

| Project | URL | Pinned commit | License | Purpose |
| --- | --- | --- | --- | --- |
| A3-A3U-robot-model | https://github.com/AgibotTech/A3-A3U-robot-model | `589f508ff357447c610a3f3004419035ddc8f153` | Mulan PSL v2 | Official A3 / A3 Ultra URDF + MJCF + meshes (source of truth for the robot) |
| holosoma | https://github.com/amazon-far/holosoma | `6e146b0af5d7cd8a39b8bb2ed05b977cf70445d3` (2026-08-11) | Apache-2.0 | Primary RL baseline (FastSAC/PPO; IsaacSim backend by default, MJWarp for smoke; DR, terrain, pushes). Lives in WSL2 at `~/holosoma`, not in this tree; local patch: see docs/tooling.md |
| IsaacLab | https://github.com/isaac-sim/IsaacLab | tag `v2.3.0` | BSD-3 | Conservative baseline framework; installed editable into `.venv-isaac` (cloned at `third_party/IsaacLab`) |

Reference-only projects (studied, NOT vendored):
| Project | URL | License | Why not vendored |
| --- | --- | --- | --- |
| agibot_x1_train | https://github.com/AgibotTech/agibot_x1_train | **none** | No license ⇒ code cannot be reused; conventions documented in docs/research/ |
| humanoid-gym | https://github.com/roboterax/humanoid-gym | BSD-3 (setup.py) | Legacy Isaac Gym Preview 4, Linux-only; methodology reference only |
