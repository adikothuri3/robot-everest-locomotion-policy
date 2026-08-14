---
title: A3 Ultra Locomotion + Get-up — Overview
updated: 2026-08-13
status: current
---

# Overview

**Objective:** a **smooth, stable, working locomotion policy + get-up policy** for the **AgiBot A3 Ultra** humanoid. Once both work and chain cleanly (fall → get up → walk), they get **fine-tuned on Everest-like terrain**, so the end product is a stable and smooth locomotion + get-up policy for alpine conditions. See [[baselines]] for exactly what we build on and the scope boundaries.

The Everest terrain itself comes from the sibling repo `C:\Users\Aditya\VSCode\GeologicDome` (footage → LingBot-Map reconstruction → sim terrain). This repo consumes that terrain later through the `TerrainSpec`/`TerrainPatch` interface; it does not reproduce that pipeline.

> [!info] Current phase (2026-08-13)
> Bootstrap is done and verified: A3 model validated in MuJoCo (12/12 diagnostics), Holosoma port trains on GPU in WSL2, cross-sim consistency vs Isaac passes, stability-suite floor recorded. Local wall-clock is too slow (~30 s/it), so **real training runs go to the cloud** (`scripts/cloud/train_a3_cloud.sh`). Next: cloud-train E02/E06 locomotion, then the get-up ladder (E11+). Full detail: `docs/bootstrap_report.md`.

## Roadmap

| Milestone | What "done" means | Status |
| --- | --- | --- |
| M0 Bootstrap | asset validated, training stack runs, floors recorded | done 2026-08-13 |
| M1 Locomotion | A3 walks smoothly on flat + rough (E02–E06), beats PD-stand floor on the stability suite | next — cloud runs |
| M2 Get-up | A3 rises from supine/prone/side ≥90% in sim (E11–E13, HoST recipe) | scaffold ready |
| M3 Handoff | fall → get up → walk chained in MuJoCo gate (E14) | blocked on M1+M2 |
| M4 Everest fine-tune | both policies fine-tuned on GeologicDome terrain (E10/E15) | blocked on terrain handoff |

## Map

- What we build on and why: [[baselines]]
- Every run ever made: [[experiments]] (append-only)
- Choices and their rationale: [[decisions]]
- Machine/env state: [[setup]]
- Known unknowns and risks: [[open-questions]]
- Deep detail lives in `docs/` (baseline matrix, bootstrap report, cloud training, ONNX interface, research reports) and `experiments/README.md` (the E00–E15 ladder definitions).
