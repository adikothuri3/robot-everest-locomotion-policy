---
title: Experiment Log
updated: 2026-08-13
status: current
---

# Experiment log

Append-only run log — **rows are never deleted**; failed runs with takeaways are the point. The ladder *definitions* (E00–E15, commands, prerequisites) live in `experiments/README.md`; this file records what was actually run. Judge runs against the floors in [[baselines]].

Every row: date, ladder ID, short commit hash of the code that ran, config deltas, outcome, takeaway.

| Date | ID | Commit | Config | Outcome | Takeaway |
| --- | --- | --- | --- | --- | --- |
| 2026-08-13 | E00 | 2bc3126 | upstream `g1-29dof-fast-sac`, mjwarp, 512 envs, 300 it | ran on GPU, ~3.8 GB VRAM — but first attempt on PyPI mujoco-warp 3.11.0 diverged (NaN rewards, 1e12 ang. vel., 4-step episodes) | pin mujoco_warp git `ecaef88`; PyPI 3.11.0 breaks physics for both G1 and A3 |
| 2026-08-13 | E01 | 9987ce6 | `a3-ultra-fast-sac`, 512 envs, 200 it (`logs/everest-a3/20260813_233010`) | trains on GPU (WarpBackend + CUDA graph), 0 nefc overflows, ~30 s/it wall-clock | chain is correct but local is ~30 s/it → train in the cloud, validate locally |
| 2026-08-13 | — | 9987ce6 | stability suite, PD-stand policy, full 68 scenarios | 25/68 survived; max push 0.2–0.3 m/s | the documented floor every trained policy must beat (`results/stability/pd_stand_baseline.json`) |
| 2026-08-13 | E12 | e28cc96+ | `a3-ultra-getup` (PPO) mjwarp smoke, 64 envs, 3 it | env+pose-bank+obs(93/97)+rollout all wired; PPO update NaN'd ("std >= 0") | NaN is NOT the get-up task: control run below reproduces it with pure locomotion |
| 2026-08-13 | E07-pre | e28cc96+ | `a3-ultra-ppo` (plain upstream locomotion PPO) mjwarp, 64 envs, 3 it | NaN in first PPO update, zero get-up code involved | **PPO+MJWarp is broken locally even for locomotion**; FastSAC survives → MJWarp smoke = FastSAC only, all PPO runs = IsaacSim cloud |
| 2026-08-13 | E12 | e28cc96+ | `a3-ultra-getup-fast-sac` mjwarp smoke, 64 envs, 5 it, njmax_per_env 768, real fallen starts + assist force | **5/5 iterations, checkpoint saved, clean exit** | get-up pipeline works end-to-end locally; lying contacts need njmax_per_env ≥1024 (one 857 frame); ready for Lambda (`scripts/cloud/train_a3_getup_cloud.sh`) |
