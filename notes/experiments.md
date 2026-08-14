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
