# Final walking policy — build spec

**Read this if you are picking up the A3 Ultra locomotion policy.** It is self-contained:
with `notes/overview.md` and this file you have everything needed to execute. It supersedes
`docs/locomotion_v2_plan.md` (deleted).

**Scope: walking locomotion only.** The get-up policies (`a3-ultra-rollover` +
`a3-ultra-getup`) are a separate track — see `notes/decisions.md` 2026-08-15 and
`docs/research/getup_recipes.md`. Nothing here touches them.

**Stack is unchanged:** Holosoma + FastSAC on IsaacSim/PhysX, v1's training config as the
base. No new trainer, no new simulator.

---

## 1. Where we are

**Baseline:** `checkpoints/cloud_20260814_012617-a3_ultra_fast_sac-locomotion/model_0050000.onnx`
— FastSAC, 50,000 iterations, 4096 envs, 204.8 M samples, ~21 k FPS, **2.7 h wall clock**.

It works. On the MuJoCo sim2sim gate (`docs/sim2sim_locomotion_report.md`):

| Measure | PD-stand floor | v1 |
| --- | --- | --- |
| Stability grid (68 scenarios) | 26/68 | **68/68** |
| Max recoverable push, standing | 0.2–0.3 m/s | **2.0–3.0 m/s** |
| Max recoverable push, walking | n/a | **2.5–4.0 m/s** |
| Extended showcase (41 scenarios) | — | **37/41** |

Do not throw this away. Every stage below is graded against it and promoted only if it wins.

### The three defects this spec fixes

All measured, none guessed.

| Defect | Evidence | Root cause | Fixed by |
| --- | --- | --- | --- |
| **Walks ~15% slow** | 0.43 m/s at a 0.5 command; 0.84 at 1.0. Error grows with speed. | `base_lin_vel` is in `critic_obs` but **not** `actor_obs` — the policy cannot see its own speed and must infer it from one frame of joint state. | **B**, **G** |
| **Yaw drift** | 3–6°/s uncommanded. `rew_tracking_ang_vel` finished at 0.559 against the 0.8 threshold in its own config's `nightly.metrics` (`rew_tracking_lin_vel` cleared 0.95 at 1.272). | Only yaw *rate* is commanded, with no heading term, so heading error integrates freely. `LocomotionCommand` samples a `heading` range and **never uses it**. | **A** |
| **Blind to arm motion** | 5/20 survival when a skill drives the arms; 20/20 with the arm observation channels frozen (E08c). | The `pose` reward weights the 14 arm joints at **50.0** against 0.01–5.0 for legs, pinning arms to **±0.018 rad**. Those observation channels are constant during training, so any real arm motion is off-distribution and corrupts the *leg* outputs. | **C**, **D** |

> [!note] The arm failure is not a balance problem
> Arms are 9.98 kg of 60.18 kg, but static CoM shift is ≤2.75 cm against a 14.7 cm
> forward margin — reach-forward falls at 2.75 cm while overhead falls having moved the CoM
> 0.66 cm *backward*. It is purely an out-of-distribution observation. Full evidence: E08b/E08c
> in `notes/experiments.md`.

---

## 2. Deploy-today contract (works right now, no retraining)

If a manipulation skill needs the arms **before** any of this is trained: freeze the arm
observation channels and hand the arms over.

- Zero (or clamp to ±0.018 rad — identical in practice) the 14 arm entries of `dof_pos`
  (obs `[52:66]`) and `dof_vel` (obs `[81:95]`) before inference.
- Let the skill own the 14 arm action columns.
- Reference implementation: `mask_arm_obs()` in `scripts/eval/sim2sim_arms.py`.

Measured under worst-case skills (3 Hz swings, joint-extreme slams every 0.5 s, full-range
poses resampled at 2 Hz, 6 kg per hand, rough terrain + 1.5 m/s shoves): **raw 5/20,
frozen 20/20**. Always-on costs nothing when idle (velerr 0.173 → 0.175). The same treatment
works for the 3 waist channels (1 Hz twist: falls at 8.1 s raw, survives frozen).

**Its limit, stated honestly:** it preserves *stability*, not *tracking*. A 3 Hz antiphase arm
swing still walks but yaw error goes to ~2.2 rad/s against a 0.066 baseline, because the policy
is blind to the momentum being injected. Untested: contact forces from real manipulation
(pushing/pulling a fixed object) — a different disturbance class entirely.

This is also the industry-standard architecture, not a hack: in decoupled loco-manipulation
systems the lower-body RL policy simply does not observe the arms.

---

## 3. The seven components

Ordered by value per unit of risk. Each names the exact hook — all verified against holosoma
`6e146b0` source, not inferred.

### A · Heading command → fixes yaw drift

`LocomotionCommand` (`managers/command/terms/locomotion.py`) has `command_dim = 3` and samples
`command_ranges["heading"]` that nothing reads.

- Extend `command_dim` to 4; entry 3 is the target heading.
- Each `step()`, derive the yaw-rate command from heading error instead of sampling it:
  `ω_cmd = clip(0.5 · wrap_to_pi(θ_target − θ), −1, 1)`, with
  `θ = atan2(forward_y, forward_x)` from the base quaternion. This is the standard
  `legged_gym` formulation and turns an integrating error into a regulated one.
- Add heading error as an observation term (1 dim).
- **Keep ~20% of episodes on pure yaw-rate commands** so spin-in-place does not regress.

### B · Concurrent velocity estimator → fixes the speed undershoot

*Requires a Holosoma fork* (`agents/fast_sac/fast_sac_agent.py`, `fast_sac.py`).

- Add a head that predicts base linear velocity (3 dims) from the actor's own observation.
- Supervise with MSE against `simulator.robot_root_states[:, 7:10]` rotated into the base
  frame — the same quantity `critic_obs.base_lin_vel` already uses.
- Feed **the estimate** (detached) into the actor as 3 extra inputs. Ground truth stays
  privileged, so the policy remains deployable.
- Train concurrently in one stage — no teacher/student distillation.
- **Log estimator error as its own scalar** and gate S1 on it. A confidently wrong estimate is
  worse than no estimate.

The literature is unambiguous here: policies with explicit velocity estimation track markedly
better than implicit encodings, and the characteristic failure of implicit encoding is
undershooting *large forward* commands — precisely our symptom, including the way our error
grows from −14% at 0.5 m/s to −16% at 1.0.

### C · Upper-body pose curriculum → arms become a trained-against disturbance

This is the standard recipe: start arms at a static pose, widen the sampled range toward full
range as the policy improves.

- New **command term** samples per-env arm targets and a motion type: **hold**, **sinusoid**
  (0.2–2.0 Hz), **step** (resample every 0.3–0.8 s). Hold and dynamic motion are *different*
  failure regimes — E08b showed sustained poses kill v1 while oscillation does not — so both
  must be in the distribution.
- Override the 14 arm action columns for ~80% of episodes in `_pre_physics_step`. The get-up
  extension already does exactly this shape of override for wrist scaling
  (`a3_ultra_getup.py:232`) — copy the pattern.
- **Amplitude curriculum:** 25% → 100% of each joint's range, driven by average episode length,
  same mechanism as `PenaltyCurriculum` (`managers/curriculum/terms/locomotion.py`).
- **Set the 14 arm entries of `pose_weights` from 50.0 → 0.0 in the same change.** If you skip
  this the policy is punished for arm deviation it does not control, and it will fight the
  curriculum.
- Add the *commanded* arm target as an observation term (14 dims) so the policy can
  **anticipate** the motion instead of only reacting to it.

Leave the remaining ~20% of episodes with the policy owning its arms, so it can still use them
when no skill is active.

### D · Centroidal angular momentum reward → arms start *helping*

Two terms, following the CAM multi-agent work:

- **Vertical CAM tracking:** `exp(−((k̂_z − k_z)/(1 + |k̂_z|))² / σ)`, with the reference `k̂_z`
  derived from the existing gait clock so it rewards anti-phase arm swing against leg motion.
- **Horizontal CAM damping:** `−min(0, Σ_{i∈x,y} k_i · k̇_i)` — penalises build-up of
  horizontal angular momentum, which is what perturbations inject.

Compute CAM from `env.simulator._rigid_body_pos` plus body velocities and masses; on IsaacSim
these come off `_robot.data` (`simulator/isaacsim/isaacsim.py:1038` shows the body-tensor
pattern). Reported against a fixed-arm baseline: natural arm swing at 1.3 m/s, **+23%
disturbance-recovery success**, lower vertical ground reaction moment.

Note v1's arms move ±0.018 rad — it never learned arm swing at all. This does not restore a
lost capability, it adds one we never had.

### E · Height scan (scandots) → terrain perception

**The plumbing already exists — this is mostly wiring.**

- Grid: **13 × 9** downward raycasts in the base *yaw* frame (yaw-only, per
  `_get_base_heights`' use of `quat_apply_yaw`), ~1.2 m × 0.8 m at 0.1 m spacing, centred
  0.2 m ahead of the base. That sits in the 6–8 cm resolution band the elevation-map
  literature converged on.
- Raycasting: `env.terrain_manager.get_state(<any string>)` returns the terrain term —
  `TerrainManager.get_state` literally does `del term_name` and returns the single term — which
  exposes `query_terrain_heights()` and the `warp_utils.ray_cast` path.
- Emit as its own observation group named `perception_obs`, add it to `actor_obs_keys` and
  `critic_obs_keys`, and set `use_cnn_encoder=True`. `encoder_obs_shape` **already defaults to
  `[1, 13, 9]`** — exactly this grid — and `CNNActor`/`CNNCritic` are already wired
  (`agents/fast_sac/fast_sac.py:158`).
- **Noise and dropout are mandatory.** Randomise per-point height noise and drop a fraction of
  points. A policy trained on a perfect height map degrades badly on a real one, and on
  hardware this comes from the head LiDAR's elevation map, which is neither perfect nor
  complete.

> [!warning] Odometry does not go into the policy
> Speed is fixed by **B** (a learned estimator), drift by **A** (a heading command), terrain by
> **E** (a height scan). Absolute position/odometry is deliberately kept *out* of locomotion
> policies across the industry — a velocity-tracking policy that consumes world position
> becomes world-dependent and breaks sim-to-real. Position error is closed by a navigation
> layer *above* the policy that re-issues velocity and heading commands.

### F · Smoothness stack

Be honest about headroom: v1's action jitter is already 0.032–0.043 with ~0% torque
saturation and the penalty curriculum fully ramped. This is polish.

- **Jerk / action curvature** — second-order difference penalty on top of the existing
  first-order `penalty_action_rate`. Penalises abrupt changes while still allowing large motion
  at constant rate.
- **DOF acceleration and torque penalties** — `penalty_dof_acc` and `penalty_torques` already
  exist in `a3_ultra_getup.py`; lift them.
- **Lipschitz-constrained policy (optional, last).** Gradient penalty `λ · ‖∇_obs a‖²` with
  **λ = 0.002**. Reported to cut action jitter 42.2 → 3.2, against 5.7 for smoothness rewards
  and 7.9 for low-pass filtering, in a few lines of code. **Caveat: validated on PPO only.**
  On FastSAC it is unproven, so it goes in as an ablation at S4 with the reward terms as the
  fallback, never as a load-bearing assumption.

### G · Observation history

`history_length` is **1** today for both `actor_obs` and `critic_obs`. Raise to **5–10**.

This is probably the highest value-per-effort item in the whole document: a history stack is
what makes implicit state estimation possible at all, it directly helps **B**, and it is a
one-line config change. It multiplies every term's width, so it changes the ONNX input
dimension and the sim2sim harness must be updated in the same commit (§5).

---

## 4. Additional changes worth making

| Change | Why | Cost |
| --- | --- | --- |
| **Terrain: set `smooth_slope` and `rough_slope` to ~0.2 each** | Both are **0.0** today (`flat 0.2 / low_obstacles 0.2 / rough 0.6`). The policy has literally never seen a sustained grade — which is exactly where the alpine scenarios fail. Fix this before blaming the Everest fine-tune. | config only |
| **100 Hz control** (`control_decimation` 4 → 2) | Doubles correction bandwidth; shows up directly as smoother tracking and better disturbance rejection. Costs sim throughput, not sample efficiency. | ~2× wall clock |
| **Widen the command envelope past ±1.0 m/s** | v1 sits at its command limit at 1.0, which is part of why tracking degrades at the top end. If you want 1.5 m/s, train it. | config only |
| **Feet air-time + landing-impact rewards** | Neither exists in Holosoma (`managers/reward/terms/locomotion.py` has `feet_phase` only). Impact penalties are what remove hard heel-strike. | 2 new terms |
| **Keep `use_symmetry=True`** | It is on today and worth preserving. New obs terms need a `mirror_obs_<term>` method — monkey-patch it onto `SymmetryUtils` from the extension. The get-up task disabled symmetry rather than do this; do not repeat that. | small |

---

## 5. Two traps that will silently waste a run

**1 · Observation order is alphabetical by term name.**
`ObservationManager.compute_group` does `sorted(obs_tensors.keys())` before `torch.cat`.
Adding `heading_error`, `upper_body_target` or `perception_obs` **reorders the vector**. The
v1 layout is:

```
actions(29) | base_ang_vel(3)×0.25 | command_ang_vel(1) | command_lin_vel(2) |
cos_phase(2) | dof_pos(29) | dof_vel(29)×0.05 | projected_gravity(3) | sin_phase(2)  = 100
```

`docs/onnx_policy_interface.md` and `src/everest_locomotion/evaluation/sim2sim.py` must be
updated **in the same commit** as any observation change. This already cost us once: the doc
had the order wrong and a correct policy scored as broken.

**2 · `use_symmetry=True` requires `mirror_obs_<name>` per term.**
The dispatch is `getattr(self, f"mirror_obs_{sub_obs_key}")` in
`agents/modules/augmentation_utils.py`. A new term without one raises at startup. For the
14-dim arm target the mirror is the same left/right swap plus sign flips that
`mirror_action_xz_plane` applies, restricted to arm indices — build it from the robot config's
`symmetry_joint_names` and `flip_sign_joint_names`.

---

## 6. Rollout

v1 cost 2.7 h. That is cheap enough that stacking every change into one run is a bad trade: if
the result is worse you cannot tell which change did it. Cost model: **0.195 s/iteration** at
4096 envs.

| Stage | Change | Gate to promote | Cost |
| --- | --- | --- | --- |
| **S0** | New extension, all features **off** | Within noise of v1: 68/68 grid, lin-vel err ≤0.16 | 10 k it · ~35 min |
| **S1** | **A** heading + **B** estimator + **G** history | Yaw drift <1°/s; speed within 5% of command at 0.5 and 1.0; `rew_tracking_ang_vel` ≥0.8; estimator error logged and converging | 50 k it · ~2.7 h |
| **S2** | **C** arm curriculum + **D** CAM | Arm suite ≥26/28 **without** observation masking; no regression on the 68-grid | 80 k it · ~4.3 h |
| **S3** | **E** height scan + slope terrain | Clears the 3 alpine combos v1 fails; rough d1.0 lin-vel err ≤0.15 | 100 k it · ~5.4 h |
| **S4** | **F** smoothness ablation (rewards vs LCP) | Jitter down ≥30% with tracking within 5% of S3 | 2 × 30 k it · ~3.2 h |

**≈16 GPU-hours total** — under one working day of Lambda, every stage attributable.

**Compressed path** if speed matters more than attribution: merge S0+S1 into one 60 k run, then
a single 120 k run with C+D+E+F together. ~11 h, two runs. The cost is real: a regression cannot
be localised without re-running the stages you skipped, which usually ends up slower.

**S0 is not optional.** It is the only thing that proves the refactor is neutral before six
behavioural changes land on top.

### Where the code goes

- New extension `src/everest_locomotion/holosoma_ext/a3_ultra_loco_v2.py`, self-aliasing into
  `sys.modules` and importing `a3_ultra_presets` — mirror the header of `a3_ultra_getup.py`.
- Register experiments `a3-ultra-fast-sac-v2` (and a `-everest` reward variant if wanted).
- Holosoma fork for **B** (and optionally **F**'s LCP) — keep it as a minimal, documented patch;
  everything else stays extension-only.
- Cloud launch via `scripts/cloud/train_a3_cloud.sh` with `EXP` overridden.

---

## 7. How to grade it

The harness exists and needs no changes for S0–S2:

```bash
P=checkpoints/<run>/model_XXXXXXX.onnx
python scripts/eval/sim2sim_suite.py --mode sweep     --run-dir checkpoints/<run>
python scripts/eval/sim2sim_suite.py --mode grid      --onnx $P --name v2 --with-baseline
python scripts/eval/sim2sim_suite.py --mode showcase  --onnx $P --name v2 --video
python scripts/eval/sim2sim_suite.py --mode pushlimit --onnx $P --name v2
python scripts/eval/sim2sim_arms.py  --onnx $P                      # arms driven by a skill
python scripts/eval/sim2sim_arms.py  --onnx $P --mask-arm-obs       # ablation
```

> [!danger] Blocking dependency before S3
> Adding scandots means the MuJoCo gate must raycast its own heightfield to build the same
> 13×9 scan, or **v2 cannot be graded at all**. `A3Sim` already has the terrain patch and
> MuJoCo exposes `mj_ray`. Build and test this during S2, not when S3 finishes.

Fill this in as stages land:

| Stage | Grid | Showcase | Push (stand/walk) | Arm suite (unmasked) | Lin-vel err | Yaw drift | Jitter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 68/68 | 37/41 | 2.0–3.0 / 2.5–4.0 | 17/28 | 0.126 | 3–6°/s | 0.032–0.043 |
| S0 | | | | | | | |
| S1 | | | | | | | |
| S2 | | | | | | | |
| S3 | | | | | | | |
| S4 | | | | | | | |

Log every run as a row in `notes/experiments.md` with its commit hash — that file is
append-only and is the project's memory.

---

## 8. Risks

- **Arm curriculum stalls.** If amplitude never widens, standing still became the cheaper
  policy. Watch amplitude against episode length; if they couple badly, drive the curriculum
  off a tracking metric instead.
- **CAM fights the pose reward.** Both act on the arms in opposite directions. The arm
  `pose_weights` must go to zero in the *same* change that adds CAM, not afterwards.
- **Scandots become a crutch.** Noise and dropout are not optional; without them the policy
  learns to trust a map hardware cannot deliver.
- **The estimator learns the wrong thing.** Gate S1 on estimator error, not just on tracking.
- **The harness falls behind the policy.** Any observation change that is not mirrored into
  `evaluation/sim2sim.py` makes a good policy look broken. This has already happened once.
- **Chasing smoothness we do not need.** v1 is already smooth. If S4 costs tracking, keep S3.

---

## 9. Recommendation

If you do only part of this, do **S0 + S1**. Heading command, velocity estimation and
observation history are small, low-risk, need no new perception, and fix the two defects
visible in the evaluation videos. That alone yields a policy that goes the speed it is told and
holds a straight line.

S2 is the arm work and is worth doing — but note it is **not required for stability under arm
motion**; §2's observation freeze already delivers 20/20 today. S2 buys *precision* during arm
motion, plus arms that actively help balance rather than merely not hurting it.
