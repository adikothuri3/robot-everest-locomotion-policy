---
title: Baselines & Scope
updated: 2026-08-18
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

## Cleared, 2026-08-14 — locomotion (E02/E06 → E08)

The first cloud run beat every locomotion floor on the MuJoCo gate. New numbers to hold,
not just beat:

| Measure | Floor | `model_0050000.onnx` |
| --- | --- | --- |
| Stability grid (68 scenarios) | 25/68 recorded, 26/68 re-measured head-to-head | **68/68** |
| Max recoverable push, standing | 0.2–0.3 m/s | **2.0–3.0 m/s** |
| Max recoverable push, walking | n/a (PD stand cannot walk) | **2.5–4.0 m/s** |
| Extended showcase (41 scenarios) | — | **37/41** |

The four failures are `friction_mu0.1` and the three *combined* alpine scenarios
(rough + slope + low friction + gusts). Each ingredient alone is survivable, which makes
the alpine fine-tune (M4) a measured gap rather than an assumed one.
Full detail: `docs/sim2sim_locomotion_report.md`; videos in `results/videos/showcase/`.

## Current locomotion policy, 2026-08-18 — v2 **S1**

`checkpoints/cloud_20260817_043529-a3_ultra_loco_v2_s1-locomotion/model_0045000.onnx`
([[decisions]], [[experiments]] E09b). **These are the numbers to hold now, not v1's.**

| Measure | v1 `model_0050000` | **S1 `model_0045000`** |
| --- | --- | --- |
| Stability grid (68 scenarios) | 68/68 | **68/68** |
| Extended showcase (41 scenarios) | 37/41 | **38/41** |
| Arm suite, both arm channels masked | 28/28 | **28/28** |
| sim2sim lin-vel error | — | **0.148** |
| sim2sim ang-vel error | — | **0.144** |
| Action jitter | — | **0.032** |
| Velocity-estimator RMS | n/a | **0.041 m/s** |

S1 clears `alpine_combo`, which v1 never has; its remaining failures are `friction_mu0.1`,
`alpine_combo_hard` and `alpine_descent`. It adds a heading command, a concurrent velocity
estimator, 5-frame observation history and scandots over v1, and widens the command
envelope to +1.5 m/s.

> [!warning] Angular tracking is still the weak axis on the *reward*
> `rew_tracking_ang_vel` finished at **0.538** for S1 against the 0.8 threshold its own
> config declares — no better than v1's 0.559, even though the heading command was added
> to fix exactly that. The sim2sim angular error *did* improve (0.406 → 0.144 over
> training), so the reward scalar and the defect it was written for have come apart.
> S3 and S4 are the first runs to clear 0.8 (0.821 / 0.843) but neither is promoted.
> Decide whether that threshold is still the right gate before the next run
> ([[open-questions]]).

> [!info] S2–S4 exist and are not promoted
> All three hold 68/68 on the grid but each failed its own gate: arm suite 13/28 (needs
> 26), alpine combos 0/3, and jitter *up* 0.069 → 0.082 where S4 promised −30%. S4 is
> still the best angular tracker in the project (0.127) and the only policy that clears
> `alpine_descent` — keep it in mind for M4, not for general walking.
