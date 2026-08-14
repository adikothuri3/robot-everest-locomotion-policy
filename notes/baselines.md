---
title: Baselines & Scope
updated: 2026-08-13
status: current
---

# Baselines & scope

What this project is built on, what it is deliberately **not**, and the measured floors every trained policy must beat. Rationale for the choices: [[decisions]] and `docs/baseline_selection.md`.

## Scope: one robot, two policies

- **Robot: AgiBot A3 Ultra only** (T2.5 locomotion variant). No G1, no H1, no cross-embodiment — G1 appears only as the upstream repro check (E00) and as the get-up recipe validation target (E12) because the recipes were proven on it. 31 actuated DOF (12 leg + 3 waist + 14 arm + 2 head), head frozen → **29 controlled**, 60.18 kg.
- **Two policies, then a chain:** locomotion (velocity tracking, rough terrain, pushes) and get-up (supine/prone/side → standing), joined by a plain switch — the get-up terminal state IS the locomotion init pose (manifest `getup.terminal`, held 2 s).
- **Everest is a fine-tune, not a rebuild:** both policies first work on generic flat/rough terrain, then get fine-tuned on GeologicDome-reconstructed alpine terrain (E10/E15). Smoothness and stability are never traded away for terrain progress.

## Training stack

- **Primary: Holosoma + FastSAC on the IsaacSim (PhysX) backend** — `simulator:isaacsim` is the registered default in the presets, `scripts/train/train_a3_wsl.sh`, and `scripts/cloud/train_a3_cloud.sh` ([[decisions]]). Its recipe already contains our stability ingredients: rough terrain + pushes + heavy DR + action-rate curriculum; hardware-validated; Apache-2.0. A3 experiments via `--import-file` presets: `a3-ultra-fast-sac`, `a3-ultra-ppo`, `a3-ultra-fast-sac-everest`.
- **Isaac Lab 2.3 + RSL-RL PPO** (native Windows, `.venv-isaac`) — same PhysX physics as the trainer: the articulation/consistency validation leg (`make check-isaac`) and the trainer fallback if Holosoma stalls.
- **MuJoCo / MJWarp — validation and smoke only.** MuJoCo classic is the independent-physics gate (stability suite, sim2sim); MJWarp is short local smoke runs only. Its pinned stack (mujoco_warp `ecaef88`) still matters for that path — see [[setup]].
- **Reference-only: AGIBOT X1 stack, Humanoid-Gym sim2sim** (legacy Isaac Gym; X1 has no license — ideas only, no code reuse).
- **Get-up recipe: HoST** (multi-critic PPO, ~350 N pull-force curriculum annealed, action-rescaler β 1.0→0.25, L2C2 smoothness), reimplemented as a Holosoma extension; HumanUP 8× slow-down as smoothness fallback. Scorecard: `docs/research/getup_recipes.md`.
- **Compute: cloud for training, local for validation.** Local FastSAC is ~11–30 s/iteration; `scripts/cloud/train_a3_cloud.sh` is fully pinned and self-verifying. Local machine runs the stability suite and sim2sim gates.

## Single sources of truth

- **Robot truth:** `configs/robots/a3_ultra.yaml` — joint order legs (L,R) → waist → arms (L,R) → head; official vs ASSUMED values (PD gains, armature, default pose) explicitly separated.
- **Training asset:** the MJCF *and* the URDF in `assets/a3_ultra/holosoma/` are both **generated** by `scripts/convert/make_holosoma_asset.py` and must stay one robot — IsaacSim imports the URDF, MuJoCo/MJWarp reads the MJCF, and Holosoma asserts the body list equals the preset's `body_names`. Full-body primitive collisions, lying keyframes, 34 bodies in both files. Never edit by hand; regenerate. `tests/test_asset_body_parity.py` fails on drift. Version history: [[decisions]].
- **Default simulator:** `isaacsim`. Anything that runs MJWarp must pass `simulator:mjwarp` (or `SIMULATOR=mjwarp`) explicitly and is a smoke run by definition.
- **Critical pin (MJWarp path only):** mujoco_warp git `ecaef88` (3.10.0) with holosoma `6e146b0`. PyPI mujoco-warp 3.11.0 **breaks physics** (NaN, 1e12 velocities) — see [[setup]].

## Measured floors (the numbers to beat)

| Floor | Value | Where recorded |
| --- | --- | --- |
| PD-stand stability suite | 25/68 scenarios survived; max recoverable push 0.2–0.3 m/s; falls at terrain difficulty ≥0.4 and friction 0.15 | `results/stability/pd_stand_baseline.json` |
| MuJoCo model diagnostics | 12/12 PASS (settle, contacts, penetration <0.1 mm, 0.15 m/s push) | `scripts/diagnostics/check_mujoco_model.py` |
| Cross-sim consistency (MuJoCo vs Isaac) | PASS when recorded; **needs re-running** — the URDF was rebuilt after that run (`make check-isaac`) | `docs/simulator_consistency.md` |
| E00/E01 training smoke | GPU WarpBackend, 0 nefc overflows, ~30 s/it locally | [[experiments]] |
