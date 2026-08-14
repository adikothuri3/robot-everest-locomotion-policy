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
| E02 | A3 flat walking, faithful upstream recipe | `exp:a3-ultra-fast-sac $IMPORT --training.num_envs 1024` (full iterations) | ready |
| E03 | + moderate DR | same as E02 (upstream DR is already on; reduce/increase via `--randomization.*` overrides) | ready |
| E04 | + push disturbances | upstream pushes are on by default; sweep `--randomization` push magnitude | ready |
| E05 | rough terrain | terrain mix is default (`terrain_locomotion_mix`); increase difficulty via terrain overrides | ready |
| E06 | Everest stability reward | `exp:a3-ultra-fast-sac-everest $IMPORT` | ready (reward variant registered) |
| E07 | PPO vs FastSAC | `exp:a3-ultra-ppo $IMPORT` vs E02 (only after both reasonably tuned) | ready |
| E08 | Cross-physics eval | export ONNX (`eval_agent.py --training.export_onnx=True`) → `scripts/eval/stability_suite.py --policy onnx` (MuJoCo classic) → later Isaac | infra ready |
| E09 | strong terrain + strong DR | E05 + widened DR ranges | needs E05 |
| E10 | first alpine curriculum | Tomasz terrain via `TerrainPatch` → custom holosoma terrain term or mesh export | blocked on terrain generator handoff |

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
| E13 | A3 flat get-up cloud run (supine/prone/side; success = holding locomotion default_pose 2 s per manifest `getup.terminal`; done when success>0.9 AND assist_scale=0 AND penalty_scale=1) | `bash scripts/cloud/train_a3_getup_cloud.sh` on Lambda (EXP=a3-ultra-getup or a3-ultra-getup-fast-sac) | ready — needs Lambda instance |
| E14 | Get-up→locomotion chained handoff in MuJoCo gate (get-up ONNX → freeze → locomotion ONNX, survive 5 s + 0.3 m/s command) | `scripts/eval/stability_suite.py` chained mode (to be added) | blocked on E13 + a trained locomotion policy (E02+) |
| E15 | Slope/rough get-up curriculum (0–15°, HiFAR DR: μ→0.1, compliance, under-body obstacles) | E13 + terrain overrides | blocked on E13 |
