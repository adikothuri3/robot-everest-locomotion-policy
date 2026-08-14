# Exported policy interface (Holosoma FastSAC/PPO → ONNX)

Verified 2026-08-14 against holosoma `6e146b0` source and a real export
(`checkpoints/cloud_20260814_012617-…/model_0050000.onnx`). The working
implementation is `src/everest_locomotion/evaluation/sim2sim.py` — prefer it over
this page if the two ever disagree.

> [!warning] The earlier version of this page was wrong
> It listed the observation in *config* order with a 1-dim gait phase. The real
> order is **alphabetical by term name** and the phase is **2-dim**. Either error
> produces a policy that falls over immediately, which is indistinguishable from
> "the checkpoint is bad" — hence this warning.

## Observation

`ObservationManager.compute_group` does `sorted(obs_tensors.keys())` before
concatenating, so the actor vector is ordered by term name, not by the order the
terms appear in `holosoma_config.yaml` (which is dumped alphabetically anyway and
therefore carries no ordering information).

For `a3_ultra_fast_sac` (history_length 1, 29 DOF → **100 dims**):

| slice | term | dims | scale | note |
| --- | --- | --- | --- | --- |
| 0:29 | `actions` | 29 | 1.0 | previous **raw** policy output (`action_manager.action`) |
| 29:32 | `base_ang_vel` | 3 | 0.25 | body frame |
| 32:33 | `command_ang_vel` | 1 | 1.0 | yaw rate |
| 33:35 | `command_lin_vel` | 2 | 1.0 | vx, vy |
| 35:37 | `cos_phase` | 2 | 1.0 | one entry per leg |
| 37:66 | `dof_pos` | 29 | 1.0 | relative to default pose, canonical order |
| 66:95 | `dof_vel` | 29 | 0.05 | |
| 95:98 | `projected_gravity` | 3 | 1.0 | |
| 98:100 | `sin_phase` | 2 | 1.0 | one entry per leg |

The whole vector is then clipped to ±`clip_observations` (100.0).

### Gait phase

`LocomotionGait` owns a `[num_envs, 2]` phase tensor. In **eval mode** the per-leg
offsets are pinned to `[0, -π]` and the frequency to `1/gait_period` (no
randomisation), and the phase advances with the control-step counter:

```
phase_k = wrap_to_pi(k * 2π * dt * gait_freq + offset)      dt = 0.02 s
```

Whenever the command is ~zero (`|v_xy| < 0.01` and `|w_z| < 0.01`) both entries are
overwritten with `stand_phase_value` = π. That override runs in `step()`, so it does
**not** apply to the reset-step observation (k = 0).

## Action

FastSAC's ONNX wrapper embeds the observation normalizer *and* the actor's per-joint
action bounds, and returns `tanh(mean) * action_scale + action_bias`. So the consumer:

* feeds **raw scaled** observations as tabulated above — do not re-normalize;
* treats the output as the raw action, and applies only the environment's uniform
  `control.action_scale` (0.25 for this robot, since
  `action_scales_by_effort_limit_over_p_gain` is false):

```
target_dof_pos = default_dof_pos + 0.25 * clip(action, ±action_clip_value)
torque         = kp * (target - q) - kd * qd,   clipped to the effort limits
```

There is no separate per-joint rescaling step to apply — the actor already carries
`max(|limit - default|) / action_scale` per joint in its exported weights.

## ONNX metadata

`attach_onnx_metadata` writes everything needed to drive the policy without the
training config: `dof_names`, `kp`, `kd`, `action_scale`, `command_ranges`,
`iteration`, `robot_urdf`, `robot_urdf_path`, and the full `experiment_config` JSON
(which is where `init_state.default_joint_angles` comes from). Read the contract from
the file rather than from a checked-in config — that is what `HolosomaPolicy` does,
and it is why a mismatched export fails loudly instead of silently.

## Consumers

1. **This repo's sim2sim gate** — `scripts/eval/sim2sim_suite.py`. MuJoCo classic,
   50 Hz, the generated training MJCF. See `docs/sim2sim_locomotion_report.md`.
2. **Holosoma's own inference stack** — `holosoma_inference/run_policy.py`
   reconstructs observations from the same metadata; it needs an inference-side
   RobotConfig for the A3 before it can be used.
