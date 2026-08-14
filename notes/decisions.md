---
title: Decisions
updated: 2026-08-13
status: current
---

# Decisions

Newest first. Each entry: what was chosen, why, what was rejected. Add an entry whenever a session makes a call that a future agent might otherwise re-litigate.

## 2026-08-13 — Get-up recipe: reimplement HoST in Holosoma

**Chosen:** HoST's recipe (multi-critic PPO, ~350 N pull-force curriculum annealed to zero, action-rescaler β 1.0→0.25, L2C2 smoothness regularization) reimplemented as a Holosoma extension, preceded by a throwaway HoST Isaac Gym feasibility probe (E11, `scripts/getup/host_probe/`) to answer "can a 60 kg A3 rise at all?" cheaply. **Fallback:** HumanUP's 8× slow-down Stage II if smoothness is the blocker. **Rejected:** running HoST's stack as-is for the real policy (legacy Isaac Gym, dead end for our pipeline). Scorecard: `docs/research/getup_recipes.md`. Key risk in [[open-questions]].

## 2026-08-13 — Train in the cloud, validate locally

**Chosen (user decision):** long training runs go to a Linux cloud GPU via `scripts/cloud/train_a3_cloud.sh` (fully pinned: holosoma `6e146b0` + mujoco_warp `ecaef88`, self-verifying). **Why:** local wall-clock is ~11–30 s/iteration (WSL2 + 4060 Ti) vs ~55 it/s upstream on a 4090 — the chain is correct, the hardware is the bottleneck. Local machine keeps the validation role: stability suite, sim2sim gates. **Rejected:** waiting out multi-day local runs; native-Linux dual-boot (untested, disruptive).

## 2026-08-13 — Pin mujoco_warp to git `ecaef88` (3.10.0)

PyPI mujoco-warp 3.11.0 produces diverging physics (NaN rewards, 1e12 angular velocities, 4-step episodes) with holosoma `6e146b0` — confirmed on both upstream G1 and A3, so it's the stack, not our port. Holosoma's conda setup pins the git commit; its uv setup installs unpinned from PyPI — that gap was the bug. Always install via `scripts/setup/wsl_pin_mjwarp.sh`; the cloud script bakes the pin in.

## 2026-08-13 — Generated primitive-collision training asset

Official A3 mesh collisions overflow MJWarp's per-env constraint budget (`nefc overflow`). **Chosen:** a generated asset (`scripts/convert/make_holosoma_asset.py`): head welded → 29 DOF, mesh collisions → AABB-fitted boxes, foot contact points added, zero-mass URDF links stamped with tiny inertials (the Isaac importer had invented ~19 kg from mesh volumes — caught by the cross-sim comparator). v5 adds full-body collision boxes + lying keyframes for get-up (v4 = solref relax). **Rule:** never hand-edit; regenerate.

## 2026-08-13 — Baseline: Holosoma + FastSAC primary, Isaac Lab fallback

**Chosen:** Holosoma + FastSAC (weighted score 4.35/5) — the recipe already contains our stability ingredients (rough terrain + pushes + heavy DR + action-rate curriculum), hardware-validated on two humanoids, Apache-2.0, and its MJWarp backend is the only modern GPU training stack that runs on this machine. **Fallback:** Isaac Lab 2.3 + RSL-RL PPO (3.75) — native Windows, aligns with the terrain work. **Reference-only:** AGIBOT X1 (no license) + Humanoid-Gym (methodology). **Rejected:** cross-embodiment checkpoints (nothing public/usable as of Aug 2026). Full matrix: `docs/baseline_selection.md`.

## 2026-08-13 — One robot: AgiBot A3 Ultra (T2.5)

The project targets the A3 Ultra exclusively; T3.0 rejected (adds only dexterous hands). A3 joint naming is string-identical to the Unitree G1 29-DOF convention (verified), which made the Holosoma port config-only — G1 remains in the loop only as upstream-repro (E00) and recipe-parity (E12) checkpoints, never as a deliverable.
