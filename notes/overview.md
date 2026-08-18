---
title: A3 Ultra Locomotion + Get-up — Overview
updated: 2026-08-18
status: current
---

# Overview

**Objective:** a **smooth, stable, working locomotion policy + get-up policy** for the **AgiBot A3 Ultra** humanoid. Once both work and chain cleanly (fall → get up → walk), they get **fine-tuned on Everest-like terrain**, so the end product is a stable and smooth locomotion + get-up policy for alpine conditions. See [[baselines]] for exactly what we build on and the scope boundaries.

The Everest terrain itself comes from the sibling repo `C:\Users\Aditya\VSCode\GeologicDome` (footage → LingBot-Map reconstruction → sim terrain). This repo consumes that terrain later through the `TerrainSpec`/`TerrainPatch` interface; it does not reproduce that pipeline.

> [!success] Current phase (2026-08-15)
> **Locomotion works.** The first Lambda run (FastSAC, IsaacSim, 4096 envs, 50 k iterations) produced a policy that walks the full command envelope, and it survives the MuJoCo sim2sim gate untuned: **68/68** on the stability grid against a **26/68** PD-stand control, recovering **2–4 m/s** shoves against a 0.2–0.3 m/s floor. Detail and videos: `docs/sim2sim_locomotion_report.md`, `results/videos/showcase/` (start with `02_vs_floor.mp4` — same push, policy vs floor, side by side), or the [summary page](https://claude.ai/code/artifact/c462411d-4103-4789-8ac2-56c37f443b0a).
>
> **The final walking policy (v2) is built and ready to launch, not yet trained.** Spec `docs/final_rl_policy.md` → implementation `src/everest_locomotion/holosoma_ext/a3_ultra_loco_v2.py` (six registered stages S0–S4 + an LCP ablation) → launcher `scripts/cloud/train_a3_v2_cloud.sh` (`bash scripts/cloud/train_a3_v2_cloud.sh s1`). All six stages smoke-run and export; the sim2sim harness now reads each policy's observation layout out of its own ONNX metadata and reproduces v1's gate exactly ([[experiments]] 2026-08-15). Also open: the get-up cloud run (E12/E13).

## Roadmap

| Milestone | What "done" means | Status |
| --- | --- | --- |
| M0 Bootstrap | asset validated, training stack runs, floors recorded | done 2026-08-13 |
| M1 Locomotion | A3 walks smoothly on flat + rough (E02–E06), beats PD-stand floor on the stability suite | done 2026-08-14 — 68/68 vs 26/68; **improved 2026-08-18 to v2 S1**, 38/41 showcase ([[baselines]]) |
| M2 Get-up | A3 rises from supine/prone/side ≥90% in sim (E11–E13, HoST recipe) | in progress — two runs wedged, cause found and fixed 2026-08-16 ([[decisions]]) |
| M3 Handoff | fall → get up → walk chained in MuJoCo gate (E14) | blocked on M2 |
| M4 Everest fine-tune | both policies fine-tuned on GeologicDome terrain (E10/E15) | blocked on terrain handoff; gap now measured, not assumed |

## Map

- What we build on and why: [[baselines]]
- Every run ever made: [[experiments]] (append-only)
- Choices and their rationale: [[decisions]]
- Machine/env state: [[setup]]
- Known unknowns and risks: [[open-questions]]
- **Building the next walking policy: `docs/final_rl_policy.md`** — the self-contained build spec (S0–S4), start there rather than re-deriving it. It is implemented in `src/everest_locomotion/holosoma_ext/a3_ultra_loco_v2.py`; run it with `scripts/cloud/train_a3_v2_cloud.sh`.
- Deep detail lives in `docs/` (baseline matrix, bootstrap report, cloud training, ONNX interface, **sim2sim locomotion report**, research reports) and `experiments/README.md` (the E00–E15 ladder definitions).
