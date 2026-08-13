# Research Report: Isaac Lab Rough-Terrain Humanoid, Cross-Embodiment, Extreme-Terrain Open Code

Researched 2026-08-13. Items marked UNVERIFIED were not confirmable from primary sources.

## A) Isaac Lab official humanoid rough-terrain locomotion

### Version / Windows / GPU
- Stable line **Isaac Lab 2.3.x** (Isaac Sim 4.5/5.0/5.1). A 3.0 Beta 2 (Isaac Sim 6.0, Newton physics) exists — unstable, avoid for now.
- **Native Windows 11 pip install: yes.** Python 3.11 venv → `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` → `isaaclab.bat --install`. Needs Windows long paths enabled, driver 580.88+ (we have 610.62).
- GPU minimum per docs: **16 GB VRAM** / 32 GB RAM. Our 4060 Ti 8GB is below spec; headless blind/height-scan locomotion with ~1024-2048 envs is widely reported workable (UNVERIFIED officially). Camera/tiled rendering will not fit.

### Isaac-Velocity-Rough-H1-v0 / G1 anatomy
- Base cfg: `source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`; H1: `.../config/h1/rough_env_cfg.py` + `agents/rsl_rl_ppo_cfg.py`; G1 analogous.
- **Obs (policy, noise-corrupted)**: base_lin_vel(±0.1), base_ang_vel(±0.2), projected_gravity(±0.05), velocity_commands, joint_pos rel(±0.01), joint_vel rel(±1.5), last actions, height_scan(±0.1, clip ±1.0). Actor/critic **symmetric** by default (only privilege = no noise at play).
- **Actions**: JointPositionActionCfg, scale 0.5, offset = default joint pos.
- **H1 reward weights**: termination -200, track_lin_vel_xy_exp +1.0, track_ang_vel_z_exp +1.0, feet_air_time +0.25, feet_slide -0.25, dof_pos_limits -1.0, joint_deviation hip/arms -0.2, torso -0.1, flat_orientation_l2 -1.0, action_rate_l2 -0.005, dof_acc_l2 -1.25e-7.
- **Events/DR**: friction randomization (static 0.8 dyn 0.6 baseline), add_base_mass ±5 kg (None for H1), reset pose/vel ranges, push_robot interval 10-15 s xy ±0.5 m/s.
- **Terminations**: 20 s time_out + base contact force > 1 N on torso/pelvis.
- **Terrain curriculum**: `terrain_levels_vel` on ROUGH_TERRAINS_CFG (pyramid slopes, stairs up/down, random rough, boxes); promote/demote on distance traveled.
- **PPO (RSL-RL)**: MLP [512,256,128] ELU, steps/env 24, lr 1e-3 adaptive KL, entropy 0.005, ~3000 iters.
- Train: `isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --headless`.

### Custom robot path
1. URDF→USD: `isaaclab -p scripts/tools/convert_urdf.py in.urdf out.usd --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none`.
2. `ArticulationCfg` with UsdFileCfg spawn, InitialStateCfg (pos, joint_pos dict), actuators dict.
3. Actuators: regex-grouped `ImplicitActuatorCfg` (stiffness/damping/effort_limit_sim/velocity_limit_sim) or DCMotorCfg/DelayedPDActuatorCfg.
4. Practical route: copy `config/g1/` folder, swap ArticulationCfg, remap joint regexes, retune deviation-reward joint lists.

## B) Cross-embodiment humanoid policies — skeptical assessment
- **H-Zero** (arXiv 2512.00971): 13 embodiments in Isaac Gym, embodiment descriptors critic-only, claims 30-min few-shot transfer. **No code, no weights. Paper-only.**
- **XHugWBC** (xhugwbc.github.io): one policy across 12 sim/7 real humanoids. "Code Coming Soon" — **unusable today**; most relevant candidate if released. Watch it.
- Others (MERL morphology transformers, get-up across morphologies): paper-only / UNVERIFIED.
- **Bottom line: no public cross-embodiment humanoid checkpoint exists to download and fine-tune (Aug 2026).** Per-robot training remains the pragmatic path; H-Zero's own numbers concede fine-tuning is cheap.

## C) Perceptive / extreme-terrain humanoid open code (2024-2026)
1. **Project Instinct / "Hiking in the Wild"** (arXiv 2601.07718): depth + proprio → joint actions, single-stage RL, foothold safety via terrain-edge detection; real full-size humanoid 2.5 m/s outdoors. **Open code**: github.com/project-instinct — InstinctLab (Isaac Lab 2.3.2/Isaac Sim 5.1 extension, 768★), InstinctMJ (mjlab/MuJoCo port), instinct_rl, instinct_onboard (ONNX deploy). G1 configs in-repo. **CC BY-NC 4.0** (research OK, products blocked). Checkpoints: not found. **Best future perception path — it is an Isaac Lab extension, so porting = new ArticulationCfg.**
2. **Humanoid-Gym / DWL**: DWL itself not fully open; humanoid-gym covered in sibling report.
3. **Humanoid Parkour Learning** (CoRL 2024, H1): humanoid version paper-only; only quadruped parkour code public.
4. **BeamDojo** (sparse footholds, G1): repo 404 — not public as of Aug 2026.
5. **PIM** (ICRA 2025): public HIMLoco repo is quadruped-only (CC BY-NC-SA). Humanoid PIM unreleased.
6. **OmniRetarget**: G1 parkour from retargeted motions; retargeting code MIT. Relevant later for skill-from-motion.
7. 2026 entries (Light-Loco-Parkour, PHP, RPL, EgoHTR): code availability UNVERIFIED/none.

**Practical ranking for this project**: (1) Isaac Lab G1-rough config as conservative blind/height-scan baseline; (2) InstinctLab later for perceptive alpine work (NC license caveat); (3) humanoid-gym only as sim2sim methodology reference. No alpine-specific pretrained checkpoint exists that is portable to A3 Ultra.
