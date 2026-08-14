---
title: A3 Ultra Locomotion + Get-up — Overview
updated: 2026-08-14
status: current
---

# Overview

**Objective:** a **smooth, stable, working locomotion policy + get-up policy** for the **AgiBot A3 Ultra** humanoid. Once both work and chain cleanly (fall → get up → walk), they get **fine-tuned on Everest-like terrain**, so the end product is a stable and smooth locomotion + get-up policy for alpine conditions. See [[baselines]] for exactly what we build on and the scope boundaries.

The Everest terrain itself comes from the sibling repo `C:\Users\Aditya\VSCode\GeologicDome` (footage → LingBot-Map reconstruction → sim terrain). This repo consumes that terrain later through the `TerrainSpec`/`TerrainPatch` interface; it does not reproduce that pipeline.

> [!success] Current phase (2026-08-14)
> **Locomotion works.** The first Lambda run (FastSAC, IsaacSim, 4096 envs, 50 k iterations) produced a policy that walks the full command envelope, and it survives the MuJoCo sim2sim gate untuned: **68/68** on the stability grid against a **26/68** PD-stand control, recovering **2–4 m/s** shoves against a 0.2–0.3 m/s floor. Detail and videos: `docs/sim2sim_locomotion_report.md`, `results/videos/showcase/` (start with `02_vs_floor.mp4` — same push, policy vs floor, side by side), or the [summary page](https://claude.ai/code/artifact/c462411d-4103-4789-8ac2-56c37f443b0a). Next: get-up cloud run (E12/E13), and a locomotion rerun with a heading term to fix the one measured weakness — uncommanded yaw drift ([[baselines]]).

## Roadmap

| Milestone | What "done" means | Status |
| --- | --- | --- |
| M0 Bootstrap | asset validated, training stack runs, floors recorded | done 2026-08-13 |
| M1 Locomotion | A3 walks smoothly on flat + rough (E02–E06), beats PD-stand floor on the stability suite | done 2026-08-14 — 68/68 vs 26/68 ([[experiments]]) |
| M2 Get-up | A3 rises from supine/prone/side ≥90% in sim (E11–E13, HoST recipe) | next — cloud run, scaffold ready |
| M3 Handoff | fall → get up → walk chained in MuJoCo gate (E14) | blocked on M2 |
| M4 Everest fine-tune | both policies fine-tuned on GeologicDome terrain (E10/E15) | blocked on terrain handoff; gap now measured, not assumed |

## Map

- What we build on and why: [[baselines]]
- Every run ever made: [[experiments]] (append-only)
- Choices and their rationale: [[decisions]]
- Machine/env state: [[setup]]
- Known unknowns and risks: [[open-questions]]
- Deep detail lives in `docs/` (baseline matrix, bootstrap report, cloud training, ONNX interface, **sim2sim locomotion report**, research reports) and `experiments/README.md` (the E00–E15 ladder definitions).
