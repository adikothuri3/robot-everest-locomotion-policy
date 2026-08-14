---
title: Decisions
updated: 2026-08-13
status: current
---

# Decisions

Newest first. Each entry: what was chosen, why, what was rejected. Add an entry whenever a session makes a call that a future agent might otherwise re-litigate.

## 2026-08-13 — Pre-cloud bug review of the get-up pipeline (two-agent audit)

**Done before spending on Lambda:** source-level verification of all 13 IsaacSim-touching code paths against pinned holosoma + IsaacLab 2.3.0 (all confirmed: reset write-then-flush, xyzw root quats, external-wrench ordering/persistence, contact forces, terrain state, limits, ONNX export, command-less operation) plus an adversarial review. **Bugs found & fixed:** (1) BLOCKER — velocity terminations fired on the PD snap at every fallen-pose reset → spawn-death loop that pins both curricula forever; fixed with a 0.5 s termination grace window (NaN catch keeps zero grace). (2) Success gate was looser than the manifest handoff contract → tightened to `getup.terminal` exactly (pelvis 0.98+, pose ≤0.30 rad, |v| ≤0.25, joint speed ≤1.0), so assist only anneals on handoff-ready stands. (3) Pose clamp used soft (0.95×) limits → hard limits (settled poses rest at hard stops). (4) Resume wiped assist-curriculum state (`reset_all` re-runs `_init_buffers` after checkpoint load) → init-once guard. (5) Stand-hold counter double-counted on reset steps → step-counter guard. (6) Pose bank now fails at import, not after the multi-minute IsaacSim scene build. (7) +1 cm spawn clearance + `update_heights()` at reset (PhysX vs MuJoCo settle geometry; step-0 assist gate). (8) Cloud script: per-algo iteration defaults (20k PPO ≠ 50k FastSAC), correct TensorBoard tag names.

## 2026-08-13 — Get-up training pipeline: Holosoma extension, single-critic adaptation

**Chosen:** get-up implemented as one `--import-file` extension (`src/everest_locomotion/holosoma_ext/a3_ultra_getup.py`): custom env `A3UltraGetupManager`, pose-bank resets as a **command term** (runs after `randomize_dof_state`, WBT precedent), assist force via `_apply_force_in_physics_step` override (per-backend dispatch mirroring `VirtualGantry`), success = holding locomotion `default_pose` 2 s, assist annealed by success-rate curriculum, smoothness penalties (action-rate/dof-vel/dof-acc/arm-weighted torque) ramped by native PenaltyCurriculum. HoST's multi-critic **dropped** (holosoma PPO is single-critic; not worth forking the algo for v1 — revisit if training stalls). `use_symmetry=False` (custom obs terms have no mirrors; side starts are asymmetric). Experiments: `a3-ultra-getup` (PPO, HoST parity, IsaacSim) + `a3-ultra-getup-fast-sac` (repo's proven algo; only variant that runs locally). Cloud: `scripts/cloud/train_a3_getup_cloud.sh` (Lambda). **Rejected:** porting HoST's Isaac Gym code (dead-end stack), motion-tracking route (no initial-pose robustness), pull-force as randomization step-term (env override is simpler and eval-safe).

## 2026-08-13 — IsaacSim/PhysX is the registered default simulator; MuJoCo demoted to validation

**Chosen (user decision):** `simulator:isaacsim` is the default everywhere — the `a3_ultra_*` presets, `scripts/train/train_a3_wsl.sh` (new `SIMULATOR` env var), the `make train-a3*` targets, and the cloud script. **Why:** holosoma's nightly matrix training-validates FastSAC on `[isaacgym, isaacsim]` only, and MJWarp NaNs deterministically under untrained-policy flailing (reproduced on the upstream G1 asset with pinned versions, so not our port). Isaac Lab 2.3 + RSL-RL stays the PhysX-side validation leg and trainer fallback — same physics engine as training now. **MuJoCo keeps two jobs:** MuJoCo classic is the independent-physics gate (stability suite, sim2sim, `check_mujoco_model.py`), and MJWarp is opt-in local smoke (`SIMULATOR=mjwarp`) — it still needs the `ecaef88` pin. **Caveat:** IsaacSim is not supported inside WSL2 on consumer setups, so locally the default path errors with instructions and real IsaacSim runs go to the cloud; `make train-a3-smoke` passes `SIMULATOR=mjwarp` explicitly. **Rejected:** promoting Isaac Lab + RSL-RL to primary *trainer* (would mean rebuilding the FastSAC recipe — pushes, DR, action-rate curriculum — as an Isaac Lab task from scratch).

## 2026-08-13 — Get-up recipe: reimplement HoST in Holosoma

**Chosen:** HoST's recipe (multi-critic PPO, ~350 N pull-force curriculum annealed to zero, action-rescaler β 1.0→0.25, L2C2 smoothness regularization) reimplemented as a Holosoma extension, preceded by a throwaway HoST Isaac Gym feasibility probe (E11, `scripts/getup/host_probe/`) to answer "can a 60 kg A3 rise at all?" cheaply. **Fallback:** HumanUP's 8× slow-down Stage II if smoothness is the blocker. **Rejected:** running HoST's stack as-is for the real policy (legacy Isaac Gym, dead end for our pipeline). Scorecard: `docs/research/getup_recipes.md`. Key risk in [[open-questions]].

## 2026-08-13 — Train in the cloud, validate locally

**Chosen (user decision):** long training runs go to a Linux cloud GPU via `scripts/cloud/train_a3_cloud.sh` (fully pinned: holosoma `6e146b0` + mujoco_warp `ecaef88`, self-verifying). **Why:** local wall-clock is ~11–30 s/iteration (WSL2 + 4060 Ti) vs ~55 it/s upstream on a 4090 — the chain is correct, the hardware is the bottleneck. Local machine keeps the validation role: stability suite, sim2sim gates. **Rejected:** waiting out multi-day local runs; native-Linux dual-boot (untested, disruptive).

## 2026-08-13 — Pin mujoco_warp to git `ecaef88` (3.10.0)

PyPI mujoco-warp 3.11.0 produces diverging physics (NaN rewards, 1e12 angular velocities, 4-step episodes) with holosoma `6e146b0` — confirmed on both upstream G1 and A3, so it's the stack, not our port. Holosoma's conda setup pins the git commit; its uv setup installs unpinned from PyPI — that gap was the bug. Always install via `scripts/setup/wsl_pin_mjwarp.sh`; the cloud script bakes the pin in.

## 2026-08-13 — Generated primitive-collision training asset

Official A3 mesh collisions overflow MJWarp's per-env constraint budget (`nefc overflow`). **Chosen:** a generated asset (`scripts/convert/make_holosoma_asset.py`): head welded → 29 DOF, mesh collisions → AABB-fitted boxes, foot contact points added, zero-mass URDF links stamped with tiny inertials (the Isaac importer had invented ~19 kg from mesh volumes — caught by the cross-sim comparator). v5 adds full-body collision boxes + lying keyframes for get-up (v4 = solref relax). **Rule:** never hand-edit; regenerate.

## 2026-08-13 — Baseline: Holosoma + FastSAC primary, Isaac Lab fallback

**Chosen:** Holosoma + FastSAC (weighted score 4.35/5) — the recipe already contains our stability ingredients (rough terrain + pushes + heavy DR + action-rate curriculum), hardware-validated on two humanoids, Apache-2.0, and its MJWarp backend was the only modern GPU training stack that ran on this machine (the *backend* half of this was superseded the same day — IsaacSim is now the default; the trainer choice stands). **Fallback:** Isaac Lab 2.3 + RSL-RL PPO (3.75) — native Windows, aligns with the terrain work. **Reference-only:** AGIBOT X1 (no license) + Humanoid-Gym (methodology). **Rejected:** cross-embodiment checkpoints (nothing public/usable as of Aug 2026). Full matrix: `docs/baseline_selection.md`.

## 2026-08-13 — One robot: AgiBot A3 Ultra (T2.5)

The project targets the A3 Ultra exclusively; T3.0 rejected (adds only dexterous hands). A3 joint naming is string-identical to the Unitree G1 29-DOF convention (verified), which made the Holosoma port config-only — G1 remains in the loop only as upstream-repro (E00) and recipe-parity (E12) checkpoints, never as a deliverable.
