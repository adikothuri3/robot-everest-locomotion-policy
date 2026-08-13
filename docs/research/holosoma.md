# Holosoma (amazon-far/holosoma) — Research Report

Researched 2026-08-13 (web + DeepWiki agent). Items marked UNVERIFIED were not confirmable from primary sources.

**Repo:** https://github.com/amazon-far/holosoma — Amazon Frontier AI & Robotics' full-stack humanoid sim-to-real RL framework (training + inference/deployment + motion retargeting). First public commit 2025-11-13.

## 1. Simulators and OS support
- **Training backends:** IsaacGym, IsaacSim (5.1.0 + IsaacLab 2.3.0), and MJWarp (MuJoCo-Warp). **MuJoCo (classic)** supported for **inference/sim-to-sim evaluation only** (`MujocoSceneManager` in `holosoma_inference`).
- **OS: Linux-only for real workflows. No native Windows support.** Setup scripts target Ubuntu 22.04/24.04. WSL2 status: UNVERIFIED upstream.
- MJWarp GPU acceleration requires NVIDIA driver >= 555.58.02 (this machine: 610.62 — OK).

## 2. RL algorithms / FastSAC
- Implemented: **PPO** and **FastSAC** (`src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`).
- FastSAC = off-policy SAC on the FastTD3 recipe: massively parallel envs, distributional Q-learning, obs normalization, automatic action scaling from robot limits.
- Paper: "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes," Seo et al. (Amazon FAR), arXiv:2512.01996. Claim: full sim-to-real locomotion policy (randomized dynamics, rough terrain, pushes, action-rate curriculum) trains in ~15 min on one RTX 4090; deployed on Unitree G1 and Booster T1.
- Multi-GPU native via torchrun.

## 3. Robots out of the box
- **Unitree G1 (29-DOF)** and **Booster T1 (29-DOF)**: full support (training configs + shipped ONNX + SDK bridges).
- Unitree H1/H1-2/Go2 only in inference SDK bridge; `docs/writing-extensions.md` shows adding a go2_12dof quadruped as extension. No Fourier, no AgiBot.
- Tasks: velocity-tracking locomotion + whole-body motion tracking (WBT: G1/IsaacSim only).

## 4. Adding a robot (AgiBot A3 Ultra)
Manager-based (IsaacLab-style) architecture. Needed:
- Assets: URDF (Viser), MJCF (MuJoCo/MJWarp), USD (IsaacSim only).
- `RobotConfig` dataclass: `dof_names` (canonical joint ordering), `num_motors`, `default_dof_angles`, `motor2joint`/`joint2motor`, `motor_kp`/`motor_kd`, `default_per_joint_action_scale`.
- ObservationConfig term lists (base_ang_vel, dof_pos, actions + scales/noise/history), per-robot files under `src/holosoma/holosoma/config_values/loco/<robot>/`.
- Actions: `JointPositionActionTerm` (PD position targets).
- Extensions injectable without forking (`--import-file` / entry points).
- Difficulty: moderate — framework designed for it; work dominated by asset quality + PD gain/action-scale tuning, not plumbing.

## 5. Domain randomization & pushes
`RandomizationManager` terms: friction (static+dynamic), link-mass scaling + added base mass, base CoM shift, PD gain randomization, torque RFI (random force injection), action delay/latency buffers, push perturbations (`push_interval_s`, `max_push_vel` as base-velocity kicks). Cross-simulator DR consistency tested in `_dr_matrix.py`.

## 6. Terrain
Plane, procedural trimesh heightfields (flat, rough, slopes smooth/rough, stairs smooth/rough), custom `.obj` meshes. Terrain curriculum (mixed-type grid, `terrain_move_up_ratio`/`terrain_move_down_ratio`). The 15-minute FastSAC recipe trains on rough terrain.

## 7. Pretrained checkpoints
- In-repo ONNX: `models/loco/g1_29dof/fastsac_g1_29dof.onnx`, `models/loco/t1_29dof/ppo_t1_29dof.onnx` (Apache-2.0). **Morphology-specific** — fixed obs/action dims and joint ordering; cannot be loaded onto A3 directly.
- Checkpoints also load from wandb URIs; training auto-uploads ONNX.

## 8. License, activity, quality
- **Apache-2.0.** ~1,586 stars; last push 2026-08-13 — actively maintained.
- Quality: high — typed dataclass configs + registry, manager architecture, CI incl. cross-simulator DR matrix, Docker (incl. Jetson Thor), uv packaging, wandb.

## 9. Dependencies / hardware
- IsaacSim env: Python 3.11, torch 2.7.0+cu128, IsaacSim 5.1.0, IsaacLab 2.3.0.
- MuJoCo/MJWarp env: Python 3.10 (Ubuntu 22.04) or 3.12 (Ubuntu 24.04); uv-based setup script.
- RTX 4060 Ti 8GB: UNVERIFIED. Reference runs: 4096 envs on 24GB 4090. Plausible with reduced num_envs/replay size; Linux requirement means WSL2 on this machine.

## 10. Sim-to-real
First-class: `holosoma_inference` (`run_policy.py`), ONNX-only export, same code path for MuJoCo sim-to-sim and real robots; Jetson Thor Docker + ROS2 service layer. Real deployments: G1, T1.

## 11. Documented install commands
```bash
git clone https://github.com/amazon-far/holosoma.git
bash scripts/setup_isaacgym.sh        # conda env hsgym (legacy)
bash scripts/setup_isaacsim.sh        # conda env hssim (Ubuntu 22.04+)
bash scripts/setup_mujoco.sh          # conda env hsmujoco (MJWarp/MuJoCo)
bash scripts/setup_mujoco_via_uv.sh   # uv venv (recommended, Linux)
bash scripts/setup_inference.sh
# Train:
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:isaacgym logger:wandb --training.seed 1
# Eval / sim-to-sim:
python src/holosoma/holosoma/eval_agent.py --checkpoint=wandb://ENTITY/PROJECT/RUN_ID/CKPT
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path .../fastsac_g1_29dof.onnx --task.no-use-joystick --task.interface lo
```

## Machine-fit assessment (added by lead)
- This workstation: Windows 11 Home + WSL2 Ubuntu-24.04, RTX 4060 Ti 8GB, driver 610.62.
- The **MJWarp training backend inside WSL2 Ubuntu-24.04 (Python 3.12 path)** is the most plausible route for Holosoma here; IsaacSim-in-WSL2 is not supported by NVIDIA on consumer setups.
- Risk: 8GB VRAM → reduce num_envs; wall-clock still expected to be workable given FastSAC sample efficiency.
