# Experiment ladder

Real runs go to the cloud (`scripts/cloud/train_a3_cloud.sh`) on the default
**`simulator:isaacsim`** backend — the command cores below inherit it from the
presets, so no `simulator:` token means IsaacSim. Local smoke runs go through
WSL2 (`wsl -d Ubuntu-24.04 -- env SIMULATOR=mjwarp bash scripts/train/train_a3_wsl.sh ...`,
env `source ~/holosoma/.venv/hsmujoco/bin/activate`) and are MJWarp-only; treat
their numbers as pipeline checks, not results.
`IMPORT=--import-file /mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py`.

Guideline for this 8 GB GPU: `--training.num_envs 512–1024` (G1 smoke used 512 at
~3.8 GB VRAM). Every run records: git commit (log it in the run notes), seed,
config dump (holosoma writes `holosoma_config.yaml` per run), TensorBoard events.

| ID | Purpose | Command core | Status |
| --- | --- | --- | --- |
| E00 | Upstream repro (G1 FastSAC, validates install) | `exp:g1-29dof-fast-sac simulator:mjwarp --training.num_envs 512 --algo.config.num_learning_iterations 300` | run 2026-08-13 (see bootstrap report) |
| E01 | A3 stand (pipeline sanity: zero-velocity commands dominate early training; also PD-stand floor via stability suite) | `exp:a3-ultra-fast-sac $IMPORT --training.num_envs 512 --algo.config.num_learning_iterations 200 --algo.config.logging_interval 10 --algo.config.save_interval 200` | run 2026-08-13 (`logs/everest-a3/20260813_233010`): GPU/WarpBackend, 0 nefc overflows, ~30 s/it wall-clock on this machine |
| E02 | A3 flat walking, faithful upstream recipe | `exp:a3-ultra-fast-sac $IMPORT --training.num_envs 1024` (full iterations) | **run 2026-08-14** on Lambda: 4096 envs, 50 000 it, 204.8 M samples, ~21 k FPS (`checkpoints/cloud_20260814_012617-…`). Episode length 28.7 → 1001 steps; `rew_tracking_lin_vel` 1.272 (gate 0.95 ✓), `rew_tracking_ang_vel` 0.559 (gate 0.8 ✗) |
| E03 | + moderate DR | same as E02 (upstream DR is already on; reduce/increase via `--randomization.*` overrides) | ready |
| E04 | + push disturbances | upstream pushes are on by default; sweep `--randomization` push magnitude | ready |
| E05 | rough terrain | terrain mix is default (`terrain_locomotion_mix`); increase difficulty via terrain overrides | ready |
| E06 | Everest stability reward | `exp:a3-ultra-fast-sac-everest $IMPORT` | ready (reward variant registered) |
| E07 | PPO vs FastSAC | `exp:a3-ultra-ppo $IMPORT` vs E02 (only after both reasonably tuned) | ready |
| E08 | Cross-physics eval | ONNX is exported automatically every `save_interval` → `scripts/eval/sim2sim_suite.py` (MuJoCo classic; `--mode sweep\|showcase\|grid\|pushlimit`) → later Isaac | **run 2026-08-14 on the E02 policy: 68/68 grid vs 26/68 PD-stand control, 2.5–4.0 m/s push ceiling, 37/41 showcase.** Report: `docs/sim2sim_locomotion_report.md` |
| E09 | strong terrain + strong DR | E05 + widened DR ranges | needs E05 |
| E10 | first alpine curriculum | Tomasz terrain via `TerrainPatch` → custom holosoma terrain term or mesh export | blocked on terrain generator handoff |

### Final walking policy (E09a–E09e) — spec: `docs/final_rl_policy.md`

Staged rebuild of the locomotion policy on the same stack, fixing the three defects the gate
measured (≈15% speed undershoot, 3–6°/s yaw drift, blindness to arm motion). ~16 GPU-h total at
0.195 s/iteration with 4096 envs. Each stage is graded on the 68-scenario grid plus the arm
suite and promoted only on its gate; **S0 is not optional** — it is what proves the refactor is
neutral before six behavioural changes land on top.

| ID | Purpose | Command core | Status |
| --- | --- | --- | --- |
| E09a | S0 — new `a3-ultra-fast-sac-v2` extension, all new features OFF (refactor is neutral) | `exp:a3-ultra-fast-sac-v2 $IMPORT2 --algo.config.num_learning_iterations 10000` | ready — spec written, code not started |
| E09b | S1 — heading command + concurrent velocity estimator + obs history 5–10 | as E09a, features A/B/G enabled | needs E09a; **B needs a holosoma fork** |
| E09c | S2 — upper-body pose curriculum + centroidal angular momentum reward | as E09b + C/D | needs E09b |
| E09d | S3 — 13×9 height scan (`perception_obs`, `use_cnn_encoder=True`) + slope terrain tiles | as E09c + E | needs E09c **and** scandot support in the MuJoCo gate |
| E09e | S4 — smoothness ablation: jerk/curvature rewards vs Lipschitz penalty (λ 0.002) | two 30k runs from the E09d checkpoint | needs E09d |

## Get-up ladder (mission: smooth self-recovery; plan in `docs/research/getup_recipes.md`)

Asset v5 (full-body collision boxes + lying keyframes) and the fallen-pose bank
(`scripts/getup/generate_fallen_poses.py`) are prerequisites — done 2026-08-13.
Note for E02+: v5 adds 16 collision boxes vs the E01 asset (26 total; locomotion
keeps self-collisions off so only vs-terrain pairs grow) and hip_pitch/hip_roll
links are now contact-capable, so "hip" termination fires slightly earlier on
falls. Watch nefc counters on the first v5 locomotion run
(`contact_pairs_multiplier=16` in presets should still be ample).

Local MJWarp smoke findings (2026-08-13, this machine):
- Lying poses need a bigger constraint budget than locomotion: pass
  `--simulator.config.mujoco_warp.njmax_per_env 1024` for get-up smoke runs
  (default None overflowed at ~670; 768 still saw one 857 frame; 64 envs).
- Working smoke command (5/5 iterations + checkpoint, 2026-08-13):
  `exp:a3-ultra-getup-fast-sac simulator:mjwarp $GETUP --training.num_envs 64
  --algo.config.num_learning_iterations 5
  --simulator.config.sim.max_episode_length_s 10
  --simulator.config.mujoco_warp.njmax_per_env 1024`
- **PPO + MJWarp NaNs locally even for plain locomotion** (`a3-ultra-ppo`, 64
  envs, upstream recipe, zero get-up code: "normal expects std >= 0" in the
  first update). FastSAC survives. So local smoke of any PPO experiment is
  impossible on MJWarp — use `a3-ultra-getup-fast-sac` for smoke, IsaacSim for
  every real PPO run. Consistent with the MJWarp-not-training-safe decision.

| ID | Purpose | Command core | Status |
| --- | --- | --- | --- |
| E11 | HoST feasibility probe: can 60 kg A3 rise at all? (throwaway, cloud GPU, Isaac Gym) | see `scripts/getup/host_probe/README.md` | scaffold ready; needs cloud GPU + Isaac Gym Preview 4 |
| E12 | Get-up task implementation in Holosoma (HoST recipe adapted to single-critic: staged rewards, 350 N assist-force curriculum annealed by success rate, PenaltyCurriculum smoothness ramp, pose-bank command-term resets, wrist soft-freeze; full design notes in the extension docstring) | `GETUP=--import-file .../holosoma_ext/a3_ultra_getup.py` → `exp:a3-ultra-getup` (PPO, IsaacSim default) or `exp:a3-ultra-getup-fast-sac` | **done 2026-08-13** — config resolves in holosoma; wiring smoke-validated on MJWarp (env+bank+obs 93/97+rollout OK) |
| E13 | A3 get-up cloud run — TWO policies (see decisions 2026-08-15): `a3-ultra-rollover` (prone+sides → supine, success = multi-body face-up 1 s) then `a3-ultra-getup` (supine → handoff pose, assist annealed, up_threshold 0.45) | `bash scripts/cloud/train_a3_getup_cloud.sh` (trains both in sequence; EXPS override for subsets; getup can warm-start run #1's checkpoint via --training.checkpoint) | **run #1 (single policy, all postures) plateaued at 30% ≈ supine share — architecture split per literature**; two-policy scripts ready |
| E14 | Get-up→locomotion chained handoff in MuJoCo gate (get-up ONNX → freeze → locomotion ONNX, survive 5 s + 0.3 m/s command) | `scripts/eval/sim2sim_suite.py` chained mode (to be added — it already runs the locomotion half) | blocked on E13 only; the locomotion policy exists as of 2026-08-14 |
| E15 | Slope/rough get-up curriculum (0–15°, HiFAR DR: μ→0.1, compliance, under-body obstacles) | E13 + terrain overrides | blocked on E13 |
