# E08 — Sim2sim evaluation of the first cloud-trained locomotion policy

**Policy:** `checkpoints/cloud_20260814_012617-a3_ultra_fast_sac-locomotion/model_0050000.onnx`
(Holosoma FastSAC, 50,000 iterations, 4096 envs, IsaacSim/PhysX, 204.8 M samples, ~21 k FPS).
**Gate:** MuJoCo classic 3.11.0 — different physics engine, same robot.
**Harness:** `scripts/eval/sim2sim_suite.py` + `src/everest_locomotion/evaluation/sim2sim.py`.
**Videos:** `results/videos/showcase/` — 41 per-scenario clips, `00_montage.mp4` (the
highlight cut), `01_highlight_web.mp4` (small, for sharing) and `02_vs_floor.mp4` (the
same 1 m/s push run side by side against the PD-stand floor). All real time: frames are
captured every 2nd control step and written at 25 fps.
**Shareable summary page:** <https://claude.ai/code/artifact/c462411d-4103-4789-8ac2-56c37f443b0a>
(private by default; a self-contained local copy with the video embedded is
`results/videos/showcase/report.html`).

## Headline

| Measure | PD-stand floor | This policy |
| --- | --- | --- |
| Stability grid (68 scenarios) | 25/68 recorded, **26/68** re-measured here | **68/68** |
| Max recoverable push, standing | 0.2–0.3 m/s | **2.0–3.0 m/s** |
| Max recoverable push, walking | n/a (cannot walk) | **2.5–4.0 m/s** |
| Extended showcase (41 scenarios) | — | **37/41** |

The PD-stand control was re-run through *this* harness on *this* asset and landed on
26/68 with a 0.2–0.3 m/s push ceiling — within one scenario of the number recorded in
[`notes/baselines.md`](../notes/baselines.md). That agreement is the harness's own
validation: the comparison is head-to-head, not against a differently-measured floor.

## What was actually reproduced

Sim2sim is only meaningful if the policy sees the same interface it trained against.
Every element below was read out of holosoma `6e146b0` source in WSL, not from docs:

* **Observation order is alphabetical by term name**, not config order —
  `ObservationManager.compute_group` sorts before concatenating. The actual 100-dim
  actor vector is `actions(29) | base_ang_vel(3)×0.25 | command_ang_vel(1) |
  command_lin_vel(2) | cos_phase(2) | dof_pos(29) | dof_vel(29)×0.05 |
  projected_gravity(3) | sin_phase(2)`.
  `docs/onnx_policy_interface.md` guessed a different order and a 1-dim phase; both were
  wrong, and either would have produced a policy that falls immediately.
* **Gait phase is per-leg (2 values).** In eval mode the offsets are pinned to `[0, -π]`
  and the frequency to `1/gait_period`; the phase is forced to `π` on both legs whenever
  the command is ~zero.
* **The ONNX embeds the observation normalizer and the per-joint action bounds**, so its
  output is the raw action and the environment applies
  `target = default_pose + 0.25 × action`, then PD with the metadata's kp/kd
  (identical to `configs/robots/a3_ultra.yaml`), torque-clipped to the effort limits.
* **Rates:** 50 Hz policy. Holosoma runs 200 Hz sim / decimation 4; MuJoCo here runs the
  MJCF's 1 ms timestep with decimation 20 — the manifest's documented evaluation stack
  (finer substep, same policy rate).

One genuine physics difference remains and is *not* corrected: the MJCF carries the
official model's passive joint damping (0.5–2.0 N·m·s/rad) and frictionloss
(0.1–2.4 N·m), which PhysX did not have. The policy is unaffected enough to walk, so
this is left in as part of the honest gap.

## Results by scenario group

Ordered by distance from the training distribution, not alphabetically.

| group | n | survived | lin-vel err (m/s) | ang-vel err (rad/s) | action jitter | max tilt | yaw drift (°/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat | 10 | 10 | 0.126 | 0.141 | 0.032 | 5.3° | 5.9 |
| rough | 4 | 4 | 0.167 | 0.123 | 0.036 | 11.8° | 3.7 |
| push | 12 | 12 | 0.311 | 0.177 | 0.040 | 10.6° | 3.1 |
| payload | 2 | 2 | 0.129 | 0.081 | 0.035 | 3.3° | 2.2 |
| friction | 4 | 3 | 0.121 | 0.080 | 0.068 | 73.3° | 4.9 |
| slope | 6 | 6 | 0.385 | 0.269 | 0.044 | 12.6° | 5.3 |
| alpine | 3 | 0 | 0.940 | 0.840 | 0.085 | 73.3° | 13.0 |

Torque saturation never exceeds 0.3% of joint-steps in any group, and the action-clip
fraction was 0.0 for the whole training run — the policy is nowhere near its actuator
limits. Mean cost of transport 0.76.

### Smoothness

Quiet stand holds 1.07 m pelvis height at 3.0° peak tilt with action jitter 0.001 — the
policy is genuinely still when told to be still, not vibrating in place. Walking jitter is
0.032–0.043 raw-action units (≈0.008–0.011 rad of commanded joint motion per 20 ms step).
The penalty curriculum reached its maximum scale (`Env/penalty_scale` 0.5 → 1.0) during
training, so these numbers are with the smoothness penalties at full weight.

### Where it breaks

| scenario | outcome |
| --- | --- |
| `friction_mu0.1` | falls at 0.2 s — feet slide out from under it on near-ice |
| `alpine_combo` (rough d0.8 + 10° climb + µ0.4 + 1.2 m/s gusts) | falls at 8.3 s |
| `alpine_combo_hard` (rough d1.0 + 15° climb + µ0.3 + 1.5 m/s gusts) | falls at 2.3 s |
| `alpine_descent` (12° descent, rough d0.8, µ0.4 + gust) | falls at 6.7 s |

Each ingredient is survivable alone: rough d1.0 ✓, 15° climb ✓, 15° descent ✓, µ0.2 ✓,
2.0 m/s shoves ✓. Only the *combination* fails. That is exactly the gap the Everest
fine-tune (M4) exists to close, and it is now measured rather than assumed.

## Known weakness: uncommanded yaw drift

With a zero yaw-rate command the robot still yaws — 3–6°/s on flat and rough ground,
worse on slopes. This is by design, not a bug: the recipe trains yaw-*rate* tracking with
no heading term in the observation, so heading error integrates freely. It shows up
independently in the training log — `Episode/rew_tracking_ang_vel` finished at 0.559,
below the 0.8 threshold this run's own config declares in its `nightly.metrics` gate,
while `rew_tracking_lin_vel` finished at 1.272 and clears its 0.95 threshold comfortably.

Linear velocity tracking is the strength; angular is the weak axis. If straight-line
heading hold matters downstream, that needs either a heading observation term or a
heavier `tracking_ang_vel` weight on the next run.

## Can an upper-body skill move the arms?

The arms hang at the robot's sides in every clip because they are *policy-controlled*,
not idle: the `pose` reward weights the 17 waist+arm joints at **50.0** against 0.01–5.0
for the legs. A manipulation skill therefore does not add arm motion — it overrides 14 of
the policy's 29 outputs. `scripts/eval/sim2sim_arms.py` measures what that costs: a
scripted arm trajectory replaces the policy's arm targets across 7 motions × 4 contexts
(stand, walk 0.5, walk 1.0, rough walk).

**Result: 17/28 survive.** The failures are all *sustained* poses — reach-forward,
overhead, asymmetric reach. Oscillatory motion is free: ±1.0 rad shoulder swings at
1.5 Hz and continuous two-arm circling cost almost nothing.

This is **not** a balance problem. The arms are 9.98 kg of the 60.18 kg robot, but the
static CoM shifts are tiny against a 14.7 cm forward margin — reach-forward moves the CoM
2.75 cm (19% of the margin) and overhead moves it 0.66 cm *backward*, yet both fall.

The cause is the observation. Re-running with the arm `dof_pos`/`dof_vel` channels masked
to default — the arms still physically move, the policy just cannot see them — gives
**28/28**, with tilt back to 3–6° and tracking error back to baseline:

| | arms visible to the policy | arm channels masked |
| --- | --- | --- |
| survived | 17/28 | **28/28** |
| worst max tilt | 75.1° | 6.3° |
| worst lin-vel error | 0.711 | 0.227 |

The legs can carry the arm motion; the actor simply never saw a non-zero arm `dof_pos`
during training (the pose reward pinned it there), so a large value on those 28 input
dims behaves like an adversarial perturbation and corrupts the *leg* outputs. Tilt grows
smoothly with amplitude (3.6° → 7.1° → 10.9° → 14.2° → fall), which is the signature of a
progressively corrupted command rather than a mechanical limit.

**Operating limits today, without masking** (held two-arm reach):

| context | shoulder pitch only | pitch + elbow |
| --- | --- | --- |
| standing | ≤1.6 rad | ≤1.2 rad |
| walking 0.5 m/s | ≤1.6 rad | ≤0.8 rad |

Tolerance shrinks as more arm channels go off-distribution at once, which is why
pitch 1.4 alone is fine but pitch 1.4 + elbow 0.9 falls.

**What to do.** Masking the arm observation channels is free and needs no retraining, and
is the right interim contract for a skill that owns the arms. It is not a guarantee: a
masked policy is blind to the arms and can only react through body tilt, so high-momentum
motion (throwing) or a payload in the hands may still break it — those were not tested.
The principled fix is to retrain with randomized upper-body poses so those channels are
in-distribution, and to drop the arm `pose` weights well below 50 so the policy is not
punished for arm deviation in the first place. Holosoma already exposes `upper_dof_names`
and `has_upper_body_dof` for exactly this.

## Checkpoint sweep

All ten saved checkpoints (5 k → 50 k) survive the 7-scenario screen, and velocity
tracking error improves steadily with training — not strictly monotonically (40 k dips on
both axes), but with no late collapse. The final checkpoint is best on linear velocity and
within 0.011 of the best on angular, so it is the one to ship:

| iteration | 5 k | 10 k | 15 k | 20 k | 25 k | 30 k | 35 k | 40 k | 45 k | 50 k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lin-vel err | 0.235 | 0.191 | 0.180 | 0.178 | 0.168 | 0.168 | 0.160 | 0.175 | 0.159 | **0.152** |
| ang-vel err | 0.272 | 0.193 | 0.164 | 0.159 | 0.147 | 0.154 | **0.140** | 0.153 | 0.151 | 0.151 |

No overfitting, no late-training collapse, and no reason to prefer an earlier checkpoint.

## Reproducing

```bash
P=checkpoints/cloud_20260814_012617-a3_ultra_fast_sac-locomotion/model_0050000.onnx
python scripts/eval/sim2sim_suite.py --mode sweep --run-dir $(dirname $P)
python scripts/eval/sim2sim_suite.py --mode showcase  --onnx $P --name showcase --video
python scripts/eval/sim2sim_suite.py --mode grid      --onnx $P --name showcase --with-baseline
python scripts/eval/sim2sim_suite.py --mode pushlimit --onnx $P --name showcase
python scripts/eval/sim2sim_compare.py --onnx $P     # policy vs floor, side by side
```

Everything is seeded and deterministic. Results land in `results/sim2sim/`, videos in
`results/videos/showcase/`.

## What this does and does not establish

**Established.** The cloud training run produced a real locomotion policy: it tracks the
full commanded velocity envelope, survives disturbances an order of magnitude past the
PD-stand floor, and transfers across a physics-engine boundary (PhysX → MuJoCo) with no
tuning. M1 has a passing gate.

**Not established.** Nothing about hardware. The PD gains, armature and default pose in
`configs/robots/a3_ultra.yaml` are still ASSUMED values, so nothing here is a sim-to-real
claim. Terrain here is procedural, not GeologicDome-reconstructed. The get-up policy is
untouched by this evaluation, so the fall → get up → walk chain (M3) is still open.
