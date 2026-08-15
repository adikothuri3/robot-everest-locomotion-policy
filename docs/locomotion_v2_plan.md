# Locomotion v2 — training plan

Same stack (Holosoma + FastSAC on IsaacSim, v1 training config as the base), six additions
targeting the three defects the sim2sim gate measured. Shareable version:
<https://claude.ai/code/artifact/09259158-b2fa-4ed5-824c-4acac29d9cbe>

**Baseline:** `checkpoints/cloud_20260814_012617-a3_ultra_fast_sac-locomotion`
(50,000 it · 4096 envs · 204.8 M samples · **2.7 h wall clock** · 68/68 stability grid).
Detail: `sim2sim_locomotion_report.md`. Runs logged as E08/E08b/E08c in
[[experiments]].

## Why — the three measured defects

| Defect | Evidence | Root cause | Fix |
| --- | --- | --- | --- |
| Walks ~15% slow | 0.43 m/s at a 0.5 command, 0.84 at 1.0; error grows with speed | `base_lin_vel` is a **critic-only** observation — the actor must infer its own speed | B |
| Yaw drift | 3–6°/s uncommanded; `rew_tracking_ang_vel` 0.559 vs its own 0.8 gate | Only yaw *rate* is commanded, no heading term, so heading error integrates. The config samples a `heading` range that is never used | A |
| Blind to its arms | 5/20 survival with a skill driving the arms, 20/20 with the channels frozen | `pose` reward weights arms at 50, pinning them to ±0.018 rad — those channels are constant in training | C, D |

> [!warning] Odometry does not go into the locomotion policy
> Three different problems, three different standard answers, only one of which is a sensor.
> Speed under-tracking → a learned **velocity estimator** (proprioception only). Yaw drift →
> a **heading command**. Terrain ahead → a **height scan**, which is what the head LiDAR's
> elevation map feeds on hardware. Absolute position/odometry is deliberately kept *out* of
> the policy across the industry: a velocity-tracking policy that consumes world position
> becomes world-dependent and breaks sim-to-real. Position error is closed by a navigation
> layer above the policy that re-issues velocity and heading commands.

## Inventory — what Holosoma already gives us

| Component | Status | Notes |
| --- | --- | --- |
| Raycast terrain height query | **have** | `TerrainLocomotion.query_terrain_heights()` + `warp_utils.ray_cast` |
| CNN encoder for a perception group | **have** | `CNNActor`/`CNNCritic` wired to `encoder_obs_key`; `encoder_obs_shape` already `[1,13,9]` — exactly a 13×9 scan. Needs `use_cnn_encoder=True` |
| `heading` command range | **have** | Sampled but never written to the command buffer (~10 lines) |
| Penalty curriculum | **have** | `PenaltyCurriculum` on average episode length — reuse as the arm-curriculum driver |
| Upper-body DOF groups | **have** | `upper_dof_names`, `arm_dof_names`, `has_upper_body_dof` populated for the A3 |
| Scandot observation term | build | New obs term producing `perception_obs` |
| Upper-body disturbance + curriculum | build | Command + curriculum term + action override (get-up extension pattern) |
| CAM reward | build | No momentum rewards exist in Holosoma |
| Jerk / curvature penalties | build | Only `penalty_action_rate` exists; `penalty_dof_acc` can be copied from the get-up extension |
| Velocity estimator head | **fork** | Extra head + loss in `FastSACAgent` |
| Lipschitz gradient penalty | **fork** | Actor-loss change; optional |

Two of ten need a Holosoma fork; the rest live in our own `--import-file` extension.

## The six components

**A · Heading command.** Add a 4th command entry (target heading) and derive the yaw-rate
command from heading error: `ω_cmd = clip(k · wrap(θ_target − θ), −1, 1)`, k ≈ 0.5, with the
heading error added to the observation. Standard `legged_gym` formulation; converts an
integrating error into a regulated one. Keep a fraction of episodes on pure yaw-rate commands
so spin-in-place still works.

**B · Concurrent velocity estimator.** A head predicting base linear velocity from the actor's
observation history, MSE-supervised against simulator truth, with *the estimate* fed to the
actor as 3 extra inputs. Trained concurrently — no teacher/student. The literature is
unambiguous that explicit velocity estimation beats implicit encoding, and that the
characteristic failure is undershooting large forward commands, which is exactly our symptom.
Ground truth stays privileged, so the policy remains deployable.

**C · Upper-body pose curriculum.** A command term samples per-env arm targets and overrides
the policy's arm actions for ~80% of episodes. Motion types hold / sinusoid / step — sustained
poses and dynamic motion are different failure regimes (E08b). Amplitude starts near 25% of
joint range and widens with average episode length. **Drop the arm entries of the `pose`
reward from 50 → 0** in the same change, or the policy is punished for deviation it does not
control. Feed the commanded arm target into the observation so the policy can anticipate
rather than react.

**D · Centroidal angular momentum reward.** Two terms: vertical CAM tracking against a
reference derived from the gait clock (produces anti-phase arm swing), and horizontal CAM
damping `−min(0, Σ k·k̇)` over x,y. Reported gains vs a fixed-arm baseline: natural arm swing
at 1.3 m/s, **+23% disturbance-recovery success**, lower vertical ground reaction moment.
This is what turns the arms from a disturbance into a balance resource.

**E · Height scan (scandots).** 13×9 downward raycasts in the base yaw frame, centred ~0.2 m
ahead, ~1.2 m × 0.8 m at 0.1 m spacing — in the 6–8 cm band the elevation-map literature
converged on, and matching the CNN encoder's existing shape. Emit as `perception_obs`, set
`use_cnn_encoder=True`, add to actor and critic keys. **Randomise scan noise and drop out a
fraction of points** — a policy trained on a perfect map degrades badly on a real one.

**F · Smoothness stack.** Current jitter is already low (0.032–0.043, torque saturation ≈0),
so this is polish, not rescue. Jerk / action-curvature (second-order) penalty on top of the
existing first-order action rate; DOF-acceleration and torque penalties copied from the get-up
extension; and optionally a Lipschitz gradient penalty `λ·‖∇_obs a‖²` with λ ≈ 0.002 (reported
to cut jitter 42.2 → 3.2 vs 5.7 for smoothness rewards and 7.9 for low-pass filtering).
**LCP is validated on PPO only** — on FastSAC it is unproven, so it goes last as an ablation
with the reward terms as fallback.

## Staged rollout

v1 cost 2.7 h. Stacking six changes into one run is a bad trade: if v2 is worse we would not
know which change did it. Each stage is a separate run graded on the 68-scenario grid plus the
arm suite, promoted only if it clears its gate.

| Stage | Change | Gate | Cost |
| --- | --- | --- | --- |
| S0 | Reproduce v1 on the new extension, features off | within noise of v1: 68/68, lin-vel err ≤0.16 | 10k it · ~35 min |
| S1 | Heading command + velocity estimator | yaw drift <1°/s; speed within 5% at 0.5 and 1.0; `rew_tracking_ang_vel` ≥0.8 | 50k it · ~2.7 h |
| S2 | Upper-body curriculum + CAM reward | arm suite ≥26/28 **without** obs masking; no grid regression | 80k it · ~4.3 h |
| S3 | Height scan | clears the 3 alpine combos v1 fails; rough d1.0 lin-vel err ≤0.15 | 100k it · ~5.4 h |
| S4 | Smoothness ablation (rewards vs LCP) | jitter −30% with tracking within 5% of S3 | 2 × 30k it · ~3.2 h |

Total ≈ **16 GPU-hours** — under one working day of Lambda time, every stage attributable.

## Further smoothness suggestions

- **100 Hz control** (decimation 4 → 2): twice the correction bandwidth. Costs sim
  throughput, not sample efficiency.
- **Observation history** — `history_length` is **1** today. A 5–10 frame stack is standard,
  makes implicit state estimation possible at all, and helps B. Probably the highest
  value-per-effort item on this page.
- **Widen the command envelope** past ±1.0 m/s — v1 is at its command limit at 1.0, which is
  part of why tracking degrades at the top end.
- **Feet air-time and landing-impact rewards** — neither exists in Holosoma; impact penalties
  are what remove the hard heel-strike.
- **Slope tiles**: `smooth_slope` and `rough_slope` are both **0.0** in the current terrain
  mix. The policy has never seen a sustained grade — which is exactly where the alpine
  scenarios fail. Set both to ~0.2 before blaming the Everest fine-tune.
- **Keep `use_symmetry` on.** New obs terms need a matching `mirror_obs_<term>` on
  `SymmetryUtils`, which the extension can monkey-patch. The get-up task disabled symmetry
  instead; don't repeat that.

## Risks

- **Arm curriculum stalls** — if amplitude never widens, standing still became cheaper. Watch
  amplitude vs episode length; if coupled badly, drive the curriculum off a tracking metric.
- **Scandots become a crutch** — noise and dropout are not optional.
- **CAM fights the pose reward** — arm pose weights must go to zero in the *same* change that
  adds CAM.
- **Estimator learns the wrong thing** — a confidently wrong velocity estimate is worse than
  none. Log estimator error separately and gate S1 on it.
- **The sim2sim harness must grow with the policy** — scandots mean the MuJoCo gate has to
  raycast the heightfield too, or v2 cannot be graded. Build that *before* S3.

## Recommendation

Do **S1 plus observation history** first: small, low-risk, targets the two defects visible in
the videos, needs no new perception. That alone gives a policy that goes the speed you ask for
and holds a straight line.

S2 is worth doing but is not required for *stability* under arm motion — the observation
freeze already gives 20/20 (E08c). S2 buys **precision** during arm motion, plus arms that
help balance rather than merely not hurt it.
