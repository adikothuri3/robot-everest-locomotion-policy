---
title: Decisions
updated: 2026-08-18
status: current
---

# Decisions

Newest first. Each entry: what was chosen, why, what was rejected. Add an entry whenever a session makes a call that a future agent might otherwise re-litigate.

## 2026-08-18 — Yaw tracking is a reward-shape bug, not a heading-observation bug (T1)

**Chosen:** branch **T1** off the promoted S1 — same 692/707 observation contract, resumed by
`--training.checkpoint`, 40k additional iterations — carrying five changes and nothing else:
`free_yaw_chain`, `tracking_precision`, `wide_friction`, `wide_push`, `hold_prob`. Run it with
`RESUME_FROM=<S1 model_*.pt> bash scripts/cloud/train_a3_v2_cloud.sh t1`.

**What the measurement showed.** `RolloutResult.heading_drift_deg` had been computed since the
first sim2sim run and reported *nowhere* — not printed, not summarised, not gated. Reading it out
of the stored S1 rows (`results/sim2sim/v2b-s1.json`) inverts the picture we had:

| clean | deg/s | | disturbed | deg/s |
| --- | --- | --- | --- | --- |
| `flat_stand` | 0.06 | | `push_right_1.5` | **−9.9** |
| `flat_walk_fwd_0.5` | −0.41 | | `slope_up_10deg` | **+7.2** |
| `flat_walk_fwd_1.0` | 0.70 | | `friction_mu0.1` | **+15.3** |
| `rough_d0.5` | −1.48 | | `flat_turn_in_place_1.0` | **−31.7** |

Straight-line flat walking is already near-perfect. The drift is entirely in the intense cases,
and `flat_turn_in_place_1.0` is the tell: against a commanded 57.3 deg/s it sheds 31.7, i.e.
**the policy turns at ~45% of the rate it is told to**.

**Root cause: `pose`.** Mapping the preset's `pose_weights` onto the canonical joint order, every
joint that produces yaw is pinned and every joint that produces forward walking is free —
`hip_yaw` **5.0**, `waist_yaw` **50.0** (as hard as the arms) against `hip_pitch` / `knee`
**0.01**. At weight −0.5 with the penalty curriculum fully ramped, a turn is ~500x more expensive
than going straight. A disturbance yaw is then doubly bad: the correction it needs *is* the motion
it is most penalised for. `free_yaw_chain` drops both to 1.0 and leaves `waist_roll`/`waist_pitch`
at 50.0 (torso upright is genuinely wanted) and the arms untouched (S1's 28/28 masked arm contract
depends on them).

**Why the heading command did not already fix it.** Two independent reasons, both measured:

1. **The gate never exercised it.** Zero of the 31 showcase scenarios set `heading`, so every
   grid, showcase and video graded S1 in the pure yaw-rate mode it trains on ~16% of the time,
   with `heading_error` pinned at 0. The new `tracking` group (10 scenarios) closes this.
2. **Closing the loop makes it *worse*, not better.** Graded with regulation on, S1 drifts
   **−15.1 deg/s** on `hold_line_push_right` against −9.9 without it, and −16.8 on
   `turn_to_heading_180`. The outer loop correctly asks for a correction the policy cannot
   deliver. That is direct evidence the deficit is actuation-side, which is what makes
   `free_yaw_chain` the primary fix rather than more heading machinery.

**Rejected:** raising `tracking_ang_vel`'s weight. The term sits at 92% of its maximum at S1's
current error (sigma 0.25 on a squared error scores a 0.1 miss at 0.96) — scaling a saturated
term scales no gradient. `tracking_precision` instead adds fine companions at sigma **0.02**,
which carry **2.3–3.2x** the gradient at the operating point, plus a `tracking_heading` term
because the P-controller's outer loop had no gradient at all and therefore settles at a permanent
`bias / gain` heading offset for free.

**The trade we are accepting.** The three new terms add **+1.124/step** of positive reward against
S1's measured net **+0.164/step**, so the penalty terms lose relative pull and the policy is being
invited to spend smoothness to buy precision. That is intended, but it is exactly how S4 regressed
(jitter 0.069 → 0.082), so **jitter ≤ 0.035 is a hard promotion gate**, not a nice-to-have. All
three terms are `exp()` forms bounded in [0, 1], so the unbounded-magnitude failure mode of the
first v2 ladder cannot recur here.

**Also folded in, because the same measurements exposed them:**

- **Friction has never been below mu 0.5.** `randomize_friction_startup` draws `U[0.5, 1.25]`
  while `friction_mu0.1` falls in 2.0 s, `mu0.2` drifts 3.3 deg/s and every alpine scenario sits
  at mu 0.3–0.4. Now `log_uniform[0.08, 1.5]` — log rather than uniform because the interesting
  decade is the bottom one (~31% of draws below 0.2 against ~8% for uniform). **We had been
  reading a domain-randomisation hole as a terrain problem.**
- **Pushes were fixed at 1.0 m/s** while the gate probes 1–4. Now `[2.0, 2.0]` per axis.
- **Heading episodes almost never start near zero error.** Upstream draws the target from
  `U(−pi, pi)` independently of current yaw, so an episode begins ~90° off and lives clipped at
  `heading_clip` — the policy practises *turning* and hardly ever practises *holding a line under
  disturbance*, which is the behaviour we ship. `hold_prob` 0.5 targets the current heading
  instead. Stand envs are always hold, and they are now regulated at all: `heading_mode` excluded
  them, leaving a fifth of every batch with no yaw regulation.
- `heading_gain` 0.5 → 1.0, halving the steady-state offset.

**Gates to promote T1** (graded on the new `tracking` group plus the existing suite):

| measure | S1 today | T1 must reach |
| --- | --- | --- |
| `flat_turn_in_place_1.0` drift | −31.7 deg/s | \|drift\| ≤ 5 |
| `hold_line_push_right` drift | −15.1 deg/s | \|drift\| ≤ 2 |
| every `hold_line_*` drift | up to −15.1 | \|drift\| ≤ 1.5 |
| `hold_line_1.5` lin-vel err | 0.368 | ≤ 0.15 |
| mean lin-vel err | 0.148 | ≤ 0.10 |
| `friction_mu0.1` / `mu0.2` | fall / survive | both survive |
| stability grid | 68/68 | 68/68 |
| action jitter | 0.032 | ≤ 0.035 |

## 2026-08-18 — Upper-body skills are a physics disturbance, never an action override (user decision)

**Chosen:** every policy from here on keeps **all 29 action channels**. An upper-body skill is injected as an external disturbance at the actuation/physics level — the policy's arm targets are no longer overwritten in `_pre_physics_step`. The policy owns its arms and must *reject* the skill's motion the way it rejects a push or a payload.

**Why the override approach has to go.** It was never a tuning problem; it removes the learning signal from 14 of 29 outputs:

1. Discarded channels are undefined channels. On ~80% of envs the arm action was thrown away before reaching the simulator, so it got no gradient. The first ladder drifted them to |a| ≈ 11 (2.9 rad) while the legs stayed at 0.45, and the exported policies scored **0/41** — they threw their arms to the limits and fell in under a second, because at export nothing overrides anything.
2. The patch worked, but only on survival. Ownership-masked `pose` + `penalty_arm_off_target` took S2–S4 from 0/41 to **68/68** on the grid and put the reward budget back in the black. It did not fix the thing component **C** exists for.
3. **The arm curriculum made arm-skill robustness worse, in every observation treatment.** The 3x4 ablation ([[experiments]] E09c) is unambiguous — S4, the only policy explicitly trained with arms driven and told their target one step early, is the worst policy in all four modes and never clears 16/28. v1, which has no arm channel at all, hits 28/28.

|  | full | none | state-hidden | target-hidden |
| --- | --- | --- | --- | --- |
| v1 (no target channel) | 17/28 | **28/28** | 28/28 | 17/28 |
| S1 (channel, no skill in training) | 14/28 | **28/28** | 19/28 | 8/28 |
| S4 (full arm curriculum) | 13/28 | 16/28 | 10/28 | 6/28 |

The diagnosis the table supports: component **C** taught a *narrow* compensation. `UpperBodyCommand` trains 0.2–2.0 Hz sinusoids and held poses; the arm suite throws 3 Hz swings and 0.5 s slams between joint extremes. S4 leans into arm motion it can predict, and that response is actively wrong for motion it cannot — worse than no response. Information was never the problem: the harness feeds the skill's real target into `upper_body_target` and S4 still scored 13/28.

**Rejected:** (a) widening `UpperBodyCommand` to match the eval suite's frequency and slams — it treats the symptom and leaves the discarded-channel defect in place, so the arm outputs still need `penalty_arm_off_target` propping them up; (b) keeping the override with a stronger regulariser — the same objection, one layer down.

**Consequences to work through when this is built:** `penalty_arm_off_target` and the ownership masks in `pose` / `penalty_action_rate` / `penalty_action_jerk` / `penalty_dof_acc` / `penalty_torques` all exist *because* channels were being discarded. With no override, the policy owns every channel and most of that machinery should be deleted rather than carried forward — but `_policy_owned_mask` returning `None` already makes them exact no-ops, so removal is cleanup, not a behaviour change. Whether `upper_body_target` stays in the observation is a separate question ([[open-questions]]).

## 2026-08-18 — Ship S1; components C, E and F are not promoted

**Chosen:** `cloud_20260817_043529-a3_ultra_loco_v2_s1/model_0045000.onnx` is the locomotion policy. It beats v1 on every measured axis and is the first policy to improve on the M1 floors:

| | v1 | **S1** |
| --- | --- | --- |
| Stability grid | 68/68 | 68/68 |
| Extended showcase | 37/41 | **38/41** (clears `alpine_combo`, which v1 never has) |
| Arm suite, both channels masked | 28/28 | **28/28** |
| sim2sim lin-vel error | — | **0.148** |
| sim2sim ang-vel error | — | **0.144** |
| action jitter | — | **0.032** |
| velocity-estimator RMS | n/a | 0.041 m/s |

**S2–S4 are not promoted, and each failed its own gate**, not a gate invented after the fact:
* **S2** — "arm suite >= 26/28 unmasked" -> **13/28**. See the decision above.
* **S3** — "clears the 3 alpine combos v1 fails" -> clears none, and adds regressions on `friction_mu0.2` and `push_front_2`. 35/41, the weakest of the four.
* **S4** — "jitter down >= 30% vs S3" -> jitter went **up**, 0.069 -> 0.082. The smoothness stack produced the jitteriest policy in the set.

S4 is not worthless: 38/41, the best angular error in the project (0.127), and the only policy that clears `alpine_descent`. It is a real candidate for the alpine work later; it is not the general-purpose walker.

**The masking contract survives for S1 and only for S1.** I expected the E08b contract to break on v2 because v2 policies are trained on `upper_body_target` — true for S4 (16/28), false for S1 (28/28), which carries the channel but never saw a skill drive it. If you mask, **mask both channels**: for S1, hiding the target alone (8/28) is worse than hiding nothing (14/28), because displaced arms with a zeroed target is an incoherent pair of inputs.

## 2026-08-18 — Never gate a curriculum on a metric that curriculum suppresses

**The bug that wedged the v3 get-up run for 14k iterations and 1.37B samples.** The action-authority curriculum (HoST's β) started at 2.0 — double the deployable action scale — and annealed only when the assist curriculum's rose proxy cleared its threshold. But β 2.0 *by itself* makes the terminal state unholdable: measured on a standing spawn with a real policy and **zero exploration noise**, mean `max|dof_vel|` over legs+waist is **1.36 rad/s** against `SUCCESS_JOINT_VEL` 1.0, so the six-way gate passes 0.2% of steps (at β 1.0: 596 consecutive steps held, jvel 0.016). β pinned `getup_success_rate` at exactly 0, the rose proxy sat in the curriculum's dead band, so β never annealed. No exit.

**Chosen:**
- **β is scheduled, never metric-gated** — `GetupAuthorityCurriculum`, linear 2.0 → 1.0 over the first 120k env steps (~5k iterations; `common_step_counter` counts env steps, and 20k iterations × 24 = 480k). Tao 2022 and HoST both schedule the rescaler; gating it was our porting error. The schedule is **one-way** (clamped to the running minimum) because `_init_counters` resets `common_step_counter` to 0 on every construction *including a resume*, while `load_checkpoint_state` restores the annealed β — a bare schedule would shove β back to 2.0 and re-wedge any resumed run.
- **The hold counter leaks instead of resetting.** Requiring 100 *strictly consecutive* steps of a six-way conjunction measures PPO's sampling noise, not the stand: at β 1.0 with converged noise the gate passes 95.5% of steps but never 100 in a row, because one of 15 leg/waist joints spikes past 1.0 rad/s every ~20 steps. Now `+1` per passing step, `−2` per failing one, floor 0. The stillness test also runs on a 0.05 s EMA of `|dof_vel|`. Still correctly unreachable for a thrashing policy (noise ≥ 0.3 never succeeds).
- **`easy_start_prob` is fixed at 0.10, decoupled from `assist_scale`.** While it was `0.05 + 0.25·assist_scale`, 30% of episodes began standing and handed `rose_rate` ~0.30 of free credit — most of the distance to the threshold the curriculum was waiting on.
- **The curriculum drives on `getup_rose_rate_fallen`** (pose-bank starts only). Both rates are logged.
- **`GetupPenaltyCurriculum`** replaces the stock one for get-up: nothing terminates in this task by design, so `average_episode_length` is pinned at the 500-step cap from iteration 0 and the penalty ramp saturated on a signal carrying no information. Now driven by the fallen-start rise rate. Rollover keeps the stock term — its episodes do terminate.

**Ruled out before landing on this,** so nobody re-checks them: actuator torque (knee 320 Nm, τ/(m·h) 3.1 vs G1's *proven* 3.0 — the A3 is better actuated per kg than the robot HoST was demonstrated on); the assist force (a standing robot holds *better* under 350 N, not worse); and gate reachability in principle (the v1 locomotion policy satisfies the handoff gate 100% of steps with 50× margin). **Reproduce any of it with `scripts/diagnostics/check_getup_terminal.py`,** which exits non-zero if the terminal state becomes unreachable again. Multi-critic remains the escalation if the fixed run plateaus (`docs/research/getup_recipes.md` pre-registered <60–70% as the trigger).

## 2026-08-18 — The `actions` observation reports the *applied* action, not the policy's raw output

`ActionManager.process_actions` stores `self._action[:] = actions` — the tensor `_pre_physics_step` handed it — and the `actions` observation term returns that buffer. So an env that rescales actions before calling `super()` (get-up does: β × wrist soft-freeze) also changes what the policy observes next step, and any eval harness must feed the applied action back.

**Verified against the trained policy, not assumed,** because the two conventions disagree sharply and picking wrong silently makes a working policy look broken. Replayed from a standing anchor, applied feedback holds **1.061 m at 0.008 rad** of pose error — matching what the run logged (`task_target_pose` per step ≈ the easy-start share, i.e. standing envs hold the default pose). Raw feedback collapses the same policy to **0.509 m and 1.366 rad**, which the telemetry rules out. `check_getup_terminal.py` re-runs this discriminator on every invocation. `A3Sim.apply_action` therefore sits on both the torque path and the observation feedback, and is identity at its defaults so every recorded locomotion grade (68/68) is unaffected.

## 2026-08-16 — An overridden action channel must still be given a target

**The bug that actually made S2–S4 unusable (0/41 on the showcase, falling in under a second).** Component **C** discards the policy's 14 arm actions on the ~80% of envs an upper-body skill owns, and `_pose_weights(arms_free=True)` zeroed the arm pose weight on *all* envs. Between them, nothing constrained those channels: no gradient from the critic where the action was thrown away, and no pose pull where it was not. They drifted.

Measured open-loop on a textbook standing observation — no simulator, no harness, so this is the policy alone:

| | arm \|a\| mean | arm \|a\| max | non-arm \|a\| mean |
| --- | --- | --- | --- |
| S1 | 0.087 | 0.190 | 0.180 |
| S2 | 5.222 | 10.876 | 0.449 |
| S3 | 8.105 | 11.479 | 0.470 |
| S4 | 7.023 | 11.486 | 1.310 |

**The legs were fine.** At `action_scale` 0.25, |a| = 11.5 is a **2.9 rad** arm deflection commanded while standing still — the exported policy throws its arms to the joint limits and falls. In training this was invisible, because the override threw the values away before they reached the simulator; at export nothing overrides them.

**Chosen, two parts:** (1) `pose` is now ownership-masked per env instead of globally released, and the arm weight is *softened* to 5.0 rather than zeroed, so the policy's own arms are still pulled toward the default pose on the envs it owns; (2) a new `penalty_arm_off_target` regresses the policy's raw arm request onto the target actually applied on skill-owned envs, giving the discarded channels a gradient and teaching "output what the arms are doing" — so when the skill detaches, the policy continues from where the arms are instead of snapping.

**Scale derived from the robot, not guessed — and the first two attempts were both wrong.** Replaying `UpperBodyCommand._sample_pose` against the real joint limits (mean usable half-range 1.84 rad, `action_scale` 0.25) gives a mean |applied| of 0.54 action units at 25% arm amplitude and 2.14 at 100%. Draft 1 used a squared error at weight −0.5: 0.50/step, the *entire* alive bonus. Draft 2 kept the square at −0.1: still 0.39/step at full amplitude (**79% of alive**), and worse, it saturated the clamp on **92%** of samples — a saturated clamp has zero gradient, so the term would have died exactly when the arms move most. Shipped as **L1 at −0.05**: 0.027/step at 25% amplitude, 0.107 at 100% (5% and 21% of alive), clamp never binds, and the gradient is constant-magnitude, which is what pinning a free-floating channel wants. These channels get zero gradient today, so a small well-scaled one is sufficient.

**Interaction worth remembering:** the action-rate mask below, applied *alone*, would have made this worse — it removes the last remaining constraint on the arm channels. The two fixes are only correct together.

**Rule:** if you override an action channel, you have taken away its learning signal. Either exclude it from the action space or give it an explicit target. Never leave it undefined and assume the override will always be there — it will not be at export.

## 2026-08-16 — Never penalise an action channel the policy does not own

**This is what broke S2, S3 and S4** ([[experiments]] E09). `_pre_physics_step` writes the upper-body skill's arm targets *into* the action vector before the action manager stores it — that is deliberate, it is how component **C** makes the arms a disturbance. But `penalty_action_rate` (weight **−2.0**, the third-largest weight in the stack) differences `action_manager.action`, so from S2 on the policy was fined for a 14-joint sinusoid it did not choose and could not reduce. Raw values: **132 (S1) → 571 (S2) → 720 (S4)**. At weight −2.0 that single term exceeded the entire `alive` budget.

**Chosen:** a `_policy_owned_mask` applied to `penalty_action_rate`, `penalty_action_jerk`, `penalty_dof_acc` and `penalty_torques` — zero on the 14 arm channels for exactly the envs where `upper_body_command.active` is true, one everywhere else. With no arm command registered the mask is `None` and all four terms reduce **exactly** to their unmasked form, so S0/S1 stay refactor-neutral (pinned in `tests/test_v2_reward_masking.py`, which asserts equality against holosoma's verbatim `sum((a_t − a_{t−1})²)`).

**Rejected:** dropping the arm channels from the action space at S2+ (changes the ONNX contract mid-ladder and forbids the policy from ever using its arms), and simply lowering the action-rate weight (would have under-penalised the legs to compensate for the arms).

**The general rule, and the reason it went unnoticed for 14 GPU-hours:** every individual term looked plausible in TensorBoard. What was never plotted was their **sum**. `only_positive_rewards=False` and `g1_29dof_termination` has no death penalty, so once net per-step reward went negative (−0.81 at S2, −2.75 at S4, against an `alive` bonus of +0.50) the optimal policy was to *fall over early* — and episode length duly fell 1000 → 801 → 797 → 754 while every reward curve still rose. `Env/net_reward_per_step` is now logged for exactly this. **Check the reward budget's sign, not just its components.**

## 2026-08-16 — S1..S4 share one observation contract so the ladder is a curriculum

**Chosen:** every stage from S1 on carries the same 692-dim actor / 707-dim critic vector (`_LADDER_OBS`), including `upper_body_target` and `height_scan` from S1, and S2+ resume the previous stage's checkpoint via `--training.checkpoint`. Iteration counts become **cumulative** (S1 50k → S2 130k → S3 230k → S4 260k) because `FastSACAgent.load` restores `global_step`.

**Why:** the first ladder trained every stage cold, because each stage widened the observation vector and `FastSACAgent.load` has no padding path — `EmpiricalNormalization.forward` raises on the shape mismatch. The cost was invisible until S4: it got **30k iterations on the hardest configuration in the ladder** and never converged (tracking still oscillating at the final checkpoint). Holding the vector fixed turns four independent runs into one curriculum, and makes S4's small budget correct rather than crippling.

**The two terms S1 does not behaviourally need are still well defined for it:** `upper_body_target` with no command term returns the policy's own last arm action in radians (information it already has via `actions`), and `height_scan` is real terrain under S1's flat/rough mix. So S1 remains the isolation test for A+B+G; it just carries the wider input. **Cost:** ~20% throughput for the CNN encoder path, and S1 must be retrained once at the new width — the current 505-dim S1 checkpoint cannot seed the chain.

**Rejected:** zero-padded weight surgery on the existing 505-dim S1 checkpoint. It would have saved the 3.3 h retrain and is mathematically clean (new columns zero-initialised ⇒ identical function at init), but it needs ~200 lines handling `Actor` vs `CNNActor`, both normalizers, optimizer state and `log_alpha`, with a cold CNN encoder injecting noise into a trained trunk — high risk of a silent degradation next to a cheap, verifiable retrain. Revisit only if GPU hours get tight.

## 2026-08-15 — Stay on T2.5 despite colleagues using T3.0 (user-confirmed)

**Chosen:** keep `a3_ultra_t2d5` as the training variant. **Why:** T3.0 is not a drop-in — +32 finger joints (63 vs 31 hinges, 65 vs 33 bodies), `head_pitch_joint` renamed, MJCF not standalone; a switch regenerates the entire generated chain (asset, pose banks, waypoints, manifest, presets, tests), invalidates the cross-sim PASS and strictly the 68/68 locomotion policy (handoff requires both policies on the same robot), and adds ~32 floppy welded hand-contact bodies exactly where get-up arm-strut contact fidelity matters. **Consistency already holds at the policy level:** the 29 trained joints are name-identical across variants, so our ONNX policies deploy on T3.0 hardware unchanged (compatibility note in the manifest). **If** the org later standardizes on T3.0, do it as a planned migration after the get-up milestone: one regeneration + full re-gate.

## 2026-08-15 — Observation history costs replay memory, not just compute (v2 OOM)

**Found on the first real S1 launch:** CUDA OOM in `algo.setup()` on a 40 GB A100, before iteration 0. `SimpleReplayBuffer` allocates four `[num_envs, buffer_size, obs_dim]` float32 tensors (obs, next_obs, critic_obs, next_critic_obs), so it scales **linearly with observation width** — and component **G** (history 1→5) multiplies that width by 5.75×. At holosoma's default 1024 slots, 4096 envs and s1's 575/590 dims that is **36.9 GiB of replay** before IsaacSim's ~7 GiB. Every history stage would have hit it (s2 41.3, s3 44.2 GiB). **Chosen:** `V2_BUFFER_SIZE = 256` for all v2 stages (s1 9.2 / s2 10.3 / s3 11.1 GiB), keeping 4096×256 = **1.05 M transitions** — a normal replay size; holosoma's 4.2 M was unusually large. S0 stays at 1024 so it remains v1 in every respect it can be. **Rejected:** dropping `num_envs` (halves throughput and changes the curriculum's `BASE_NUM_ENVS` scaling), and dropping `critic_obs` history to 1 — that is the *right* second lever if a 24 GB card is ever the target (the critic gets privileged `base_lin_vel`, so unlike the actor it does not need history to estimate state), but it is a second variable and memory did not require it. **Rule:** any observation-width change is a replay-memory change; compute the buffer footprint before launching.

## 2026-08-15 — CAM tracking must be normalised by its own scale, not an absolute constant

**Found while sanity-checking the one guessed constant before an overnight queue.** `docs/final_rl_policy.md` §3D writes the vertical-CAM reward as `exp(-((k̂_z − k_z)/(1 + |k̂_z|))²/σ)`. That denominator assumes |k| is O(1). Mass-normalised CAM for a 60 kg walker is **O(0.005) m²/s** by a limb-momentum estimate (arms ~0.7 kg·m²/s against legs ~1.1 the other way; residual ÷60 kg), so with the original `CAM_REF_AMPLITUDE = 0.05` the whole term varied only between **0.9980 and 1.0000** across a 10× range of actual k_z — a weight-1.0 constant, next to `alive` at 10, that would have shaped nothing while looking perfectly healthy in the logs. **Chosen:** normalise the error by the reference scale (`amplitude·speed + still_scale`) so `err` is O(1) at "missed by a full swing" for *any* amplitude, making σ dimensionless (0.25). Verified numerically: the reward now spans 0.0008→1.0 over the same range. `CAM_REF_AMPLITUDE` drops to 0.01 and becomes the single constant feeding both CAM terms, to be read off `Env/cam_z` on the first S2 run. **Why this is not just a better guess:** the conditioning fix holds even if the amplitude is off by 10×; only the *behaviour asked for* still depends on the number. **Rule:** a reward whose denominator is an absolute constant is only correct if you know the quantity's magnitude — normalise by the quantity's own scale when you don't.

## 2026-08-15 — Final walking policy v2: extension-only, no Holosoma fork

**Chosen:** the whole S0–S4 ladder of `docs/final_rl_policy.md` ships as **one `--import-file` extension**, `src/everest_locomotion/holosoma_ext/a3_ultra_loco_v2.py`, with `scripts/cloud/train_a3_v2_cloud.sh` as the Lambda launcher. **Why the fork was avoidable:** the spec assumed component **B** (concurrent velocity estimator) required patching `agents/fast_sac/`, but `FastSACAlgoConfig._target_` is a plain dotted path resolved by `train_agent.get_class`, so a `FastSACAgent` subclass declared *in the extension* is reachable from config. The estimator head is added by wrapping `Actor`/`CNNActor` (`_setup_network_with_input_dim` + `process_obs`), so it lives **inside** the exported ONNX and composes with the CNN encoder for free; the Lipschitz ablation rides the same subclass. A fork would have meant maintaining a patch against a pinned commit for the life of the project, and would have put the estimator outside the artifact we actually ship. **Rejected:** forking holosoma (the spec's §3B plan), and an env-side estimator (would not have been exported with the policy, so the sim2sim gate could not run it).

## 2026-08-15 — `SymmetryUtils.mirror_xz_plane` is wrong for history > 1; patched in the extension

**Found:** `ObservationManager` stores history **per term** — a group is `[term_a(t-4..t) | term_b(t-4..t) | ...]` — but `SymmetryUtils.mirror_xz_plane` reshapes the whole group to `[B, history, single_frame_dim]`, which assumes frame-major storage. The two agree only at `history_length == 1`. **Consequence:** component **G** (history 1 → 5) combined with the `use_symmetry=True` that v1 relies on would have mirrored the wrong columns into every augmented sample — silently, with no error and no obvious metric. This is almost certainly why the get-up task disabled symmetry rather than fix it. **Chosen:** monkey-patch a term-major `mirror_xz_plane` onto `SymmetryUtils` at extension import (identical behaviour at history 1, so v1/S0 are unaffected), and keep `use_symmetry=True` for every stage. **Rejected:** disabling symmetry (throws away a validated 2× sample multiplier), and upstreaming first (the pin is `6e146b0`; the patch is 12 lines and local).

## 2026-08-15 — The observation layout travels with the policy, not in this repo

**Chosen:** `A3UltraFastSACAgent.export` attaches an `actor_obs_layout` blob to the ONNX (every term's name, dim, scale, history and offset, plus control period, gait period, command dim, scandot geometry and arm indices), and `evaluation/sim2sim.py` **builds the observation vector from it** instead of from a hardcoded table. Policies without it fall back to `ObsLayout.v1_default`. **Why:** `docs/final_rl_policy.md` §5 names this the trap that already cost one run — observation terms concatenate in alphabetical order, so adding `heading_error` renumbers everything after it, and a harness that disagrees makes a good policy look broken. A table in a doc has to be updated by hand; metadata cannot desynchronise from the policy that wrote it. Verified: the refactored harness reproduces v1's recorded gate exactly (**68/68**, 0.431 m/s @ 0.5, 0.842 @ 1.0, jitter 0.035/0.043). **Rejected:** versioned layout tables keyed by experiment name (same failure mode, more code).

## 2026-08-15 — Two deliberate departures from `docs/final_rl_policy.md`'s reward formulas

**CAM damping sign and scale.** The spec writes horizontal-CAM damping as `-min(0, Σ k_i·k̇_i)`, which is positive when momentum is *decaying* — with the negative weight a penalty term carries, that penalises the behaviour we want. Implemented as `relu(Σ k_i·k̇_i)` with the sign in the weight. It is also normalised by a reference CAM and clamped: raw `k·k̇` is unbounded (measured O(1e3) on a smoke run) and would swamp every other term. **Impact penalty.** Same reason — an unclamped squared contact force reached raw episode sums of ~1e5 in a smoke run; threshold moved to ~1.5× body weight (900 N) and the excess clamped. **`penalty_dof_acc`** is normalised per joint by `vel_limit/dt` rather than raw, because this robot's joint speeds span 6–23 rad/s and a raw sum is both unbalanced and unbounded. All three were caught by scale-checking a 25-iteration MJWarp smoke before spending cloud hours — cheap, and worth repeating for any new reward term.

## 2026-08-15 — Shipped policies live in git as ONNX only; repo is Apache-2.0

**Chosen:** commit the **final `.onnx` + its `holosoma_config.yaml` per run** (~1.9 MB total, force-added past the `checkpoints/` ignore), and nothing else from a training run. `.pt` states (18 MB each) and tfevents (up to 56 MB) stay out; if someone needs to resume training they get release assets on request. Repo licensed **Apache-2.0** (matches Holosoma upstream; patent grant), with `third_party/` attribution in the README — the AgiBot model (Mulan PSL v2) is cloned at setup, never redistributed here. **Why:** the repo is public, and a public git history is permanent — 670 MB of `checkpoints/` + `results/` would be un-removable once anyone clones. The ONNX is also the *only* artifact needed to test a policy: `evaluation/sim2sim.py` consumes it and shares no code with the trainer, which is what makes the gate independent. The shipped get-up ONNX is labelled **not a deliverable** (28k plateau ≈ 30%) so it can't be mistaken for a result — see the two-policy split below. **Rejected:** committing all `.onnx` per run (~23 MB of intermediate checkpoints nobody evaluates), git LFS (metered bandwidth on public repos), and full-checkpoint commits.

## 2026-08-15 — Get-up v3: assist gate corrected, KSI waypoints, authority curriculum (research round 3)

**Chosen:** (1) assist force re-gated to HoST's validated semantics — fires while trunk NEAR-VERTICAL (aids the torque-critical rise), never drags a flat-lying robot; annealed by a loose terminal-height "rose rate" (HoST's trigger) so the strict handoff contract is learned force-free. (2) KSI: 450 sit/kneel/crouch waypoint poses at 25% of getup resets (UniReLo's biggest ablation; HiFAR/H1-2 precedent). (3) Strong-to-weak action-authority curriculum 2.0→1.0 with β in obs (HoST/Tao'22/HumanUP-supported). (4) Actor history 5. (5) H1-2-style height-gated arm-support reward + stuck-low termination — the latter gated on height **and** projected gravity (`pg_z > -0.5`), because a robot that has SAT UP sits at pelvis 0.15–0.29 m and a height-only gate would recycle successful sitters mid-exploration, killing the sit→crouch transition KSI exists to densify. **Strategy verdict:** hip-sit-up→tuck→deep-squat route transfers to the A3 (HoST's G1 has NO waist pitch — waist worry void; A3 specific leg torque beats G1 and the hardware-proven ~70 kg H1-2). **Rejected with evidence:** multi-critic (ablation confounded with advantage normalization; HumanUP 95% single-critic; escalate only if plateau <60-70%), Stage-II slowed tracking (smoothness fallback only), SAC main line, kip-up route (heavy-robot momentum rise failed sim-to-real in its one precedent). Full citations: `docs/research/getup_recipes.md` update (2).

## 2026-08-15 — Get-up is TWO policies (rollover + supine get-up), not one

**Chosen:** split into `a3-ultra-rollover` (prone + side starts → settled supine, 5 s episodes, no assist force, multi-body face-up success per HumanUP, HoST-style roll-rate reward) and `a3-ultra-getup` (supine starts only + 5–30% standing anchors → manifest handoff pose, assist curriculum, `up_threshold` 0.45), chained at runtime by a projected-gravity router (tilt >37° → damp → chest-up? get-up : rollover; A→B on face-up cosine ≥0.9 held 0.5 s; B→locomotion on handoff gate held 0.5 s — FRASA's debounce, Unitree's FSM pattern). **Why:** run #1 (one policy, all postures mixed) plateaued at 30% ≈ the bank's supine share, and deep research confirmed the literature never made mixing work: HoST's own limitations section says posture mixing "negatively impacted performance due to interference between sampled rollouts"; HoST ships one policy PER posture; HumanUP ships rollover (98.3% hw) + get-up (78.3% hw). Side-lying goes to the ROLLOVER policy (HoST's side coverage was the prone policy zero-shot). Exploration lesson: nobody fixes PPO std collapse with noise floors — they make the task easy first (assist force, weak-reg discovery stage, keyframe resets). **Rejected:** UniReLo-style single policy (needs unreleased motion-prior data + 3 AMP discriminators), HiFAR staged-dimensionality single policy (no precedent above 23 kg-class), std floors/entropy hacks. Full spec: `docs/research/getup_recipes.md`.

## 2026-08-14 — Trained policies are evaluated by a new sim2sim harness, not the stability suite

**Chosen:** `src/everest_locomotion/evaluation/sim2sim.py` + `scripts/eval/sim2sim_suite.py` own all trained-policy evaluation (four modes: `sweep`, `showcase`, `grid`, `pushlimit`), and it loads the **generated training MJCF**, not the official one, so the gate tests the same robot the URDF gave IsaacSim. **Why:** the observation contract could not be expressed in `stability_suite.py`'s obs dict — the real actor vector is 100-dim, concatenated in **alphabetical term order** (`ObservationManager.compute_group` sorts before `torch.cat`) and includes a **2-dim per-leg gait clock** with eval-mode phase offsets `[0, -π]` and a stand override at `π`. `docs/onnx_policy_interface.md` documented neither, and both errors produce a policy that falls instantly — indistinguishable from a bad checkpoint. Contract re-derived from holosoma `6e146b0` source and verified numerically (gyro vs `qvel[3:6]` for the body-frame angular velocity, `mj_objectVelocity` for the rotation convention). `policies.OnnxPolicy` now **raises** instead of running, and `stability_suite.py` lost its `--policy onnx` path, so the only remaining trap is closed; the suite keeps its one real job, recording the PD-stand floor. **Rejected:** patching `OnnxPolicy` in place (its caller cannot supply a gait clock or per-step phase at all).

## 2026-08-14 — Sim2sim spawn height is searched, not assumed

Spawning at the terrain height under the *pelvis* buries the toe of a 0.27 m foot by ~2 cm on a 10° grade; MuJoCo resolves that penetration explosively and the episode dies at t≈0.3 s. That artifact alone made every slope ≥10° and all three alpine scenarios look like locomotion failures. Clearing the footprint *maximum* instead overcorrects (a 13 cm drop onto a slope is its own destabiliser), so `A3Sim._clear_spawn_height` scans upward for the lowest penetration-free height via `mj_forward`. After the fix, 10° and 15° slopes both survive and the remaining falls are late-episode (2.3–8.3 s) and genuine. **Rule:** on generated terrain, never trust a nominal spawn height — measure it.

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
