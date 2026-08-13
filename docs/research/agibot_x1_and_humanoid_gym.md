# Research Report: AGIBOT_x1_train vs. Humanoid-Gym

Researched 2026-08-13. Items marked UNVERIFIED were not confirmable from primary sources.

## A) AgibotTech/agibot_x1_train

### Simulator, framework, OS
- **Isaac Gym Preview 4** (legacy, Linux-only), Python 3.8, PyTorch 1.13 + CUDA 11.7.
- Structurally a fork of **roboterax/humanoid-gym** (itself from legged_gym + rsl_rl). RL algo vendored at `humanoid/algo/ppo/`.

### Algorithm, architecture, obs/actions, control
- **DHPPO** (`dh_ppo.py`) with `ActorCriticDH`: dual-history — short-history MLP state estimator outputs 3-dim estimated base velocity; 1D-CNN long-history encoder (kernel [6,4], filters [32,16], stride [3,2]) outputs 64-dim latent. Actor input = short obs stack + latent + vel estimate. Actor [512,256,128], critic [768,256,128], asymmetric.
- **Obs**: 47 dims/frame (5 command incl. sin/cos gait phase, 12 joint pos, 12 joint vel, 12 prev actions, 3 base ang vel, 3 base Euler), all with simulated lag buffers. frame_stack=66 long, short_frame_stack=5. Privileged 73×3 (adds lin vel, push force, friction, mass, contact masks, ref-pose error).
- **Actions**: 12 joint-position offsets, action_scale 0.5, PD 'P' control.
- **Control**: sim dt 0.001, decimation 10 → **100 Hz policy**. PD: hip p/r/y 30/40/35, knee 100, ankles 35 (damping 3/3/4/10/0.5/0.5).

### Rewards, DR, sim-to-real
- ~19-24 terms: ref_joint_pos 2.2 (sinusoidal per-leg gait reference from phase clock), tracking_lin_vel 1.8, tracking_ang_vel 1.1, feet_contact_number 2.0, stand_still 2.5, clearance/air-time/orientation/base-height, torque/vel/acc penalties, dof_pos_limits -10, collision -1.
- DR: friction [0.2,1.3], base mass ±3 kg, motor offset ±0.035 rad, PD ×[0.8,1.2], pushes every 4 s, **actuator lag randomized 5-40 timesteps + DOF obs lag 0-40** (heavier latency modeling than humanoid-gym).
- Terrain: trimesh curriculum (30% flat, 20% rough, slopes, discrete).
- Pipeline: train → play → export **JIT + ONNX** → sim2sim in MuJoCo (`scripts/sim2sim.py`) → `agibot_x1_infer` (C++ ONNX Runtime on AimRT, ROS2 Humble).

### Portability facts
- README: code "can be imported to other robot models". legged_gym pattern: URDF + MJCF + config classes.
- **X1 = 12 actuated leg joints only** (arms/waist/head fixed in URDF). Much smaller robot than A3 Ultra.
- **No license file** (sibling infer repo is Mulan PSL-2.0) — legally ambiguous for code reuse. Frozen: single code drop 2024-10-23. No pretrained checkpoints.

## B) roboterax/humanoid-gym
- Isaac Gym Preview 4 + legged_gym/rsl_rl; Python 3.8, torch 1.13, mujoco==2.3.6 for sim2sim. Linux-only.
- Signature: **Isaac Gym → MuJoCo sim2sim** pipeline; paper arXiv:2404.05695 claims zero-shot sim2real on RobotEra XBot-S/L.
- Standard PPO, MLP [512,256,128]/[768,256,128], 15-frame stack of 47-dim obs, 100 Hz, action_scale 0.25, high PD (knee 350). Gait-reference rewards, cycle_time 0.64 s. DR simpler than X1's.
- BSD-3-Clause (in setup.py/headers, no root LICENSE). Dormant (last push 2025-01). Ships one XBot JIT checkpoint (`policy_example.pt`).
- **Isaac Gym Preview 4 is officially legacy/unsupported** (NVIDIA recommends Isaac Lab); Linux-only, Python ≤3.8, ancient torch — applies equally to agibot_x1_train.

## C) AgiBot A3 / A3 Ultra training code
- **AgibotTech publishes no RL training code for A3** (org scan ~19 repos; only X1 train/infer/hardware + x2_urdf are locomotion-adjacent).
- Press on "Yuanzheng A3" claims 1.73 m / 55 kg / 51 DOF (UNVERIFIED; our official model repo shows 31 actuated DOF for A3 Ultra T2.5 without hands, 60.2 kg total in MJCF — official repo is our ground truth).

## Verdict for this project
- **agibot_x1_train**: reference for AgiBot control conventions (100 Hz, lag randomization ranges, PD-gain style, dual-history estimator idea) — NOT a runnable baseline here (Isaac Gym legacy, Linux-only, no license, 12-DOF).
- **humanoid-gym**: benchmark/reference for sim2sim methodology only; same legacy foundation.
