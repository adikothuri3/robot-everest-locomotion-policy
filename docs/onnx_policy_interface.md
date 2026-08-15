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
therefore carries no ordering information). When a group has
`history_length > 1`, each term occupies `dim * history` contiguous columns
**oldest frame first**, and terms are concatenated after that — the vector is
term-major, not frame-major.

> [!tip] v2 policies describe themselves — do not read the table below for them
> `a3_ultra_loco_v2` policies carry an **`actor_obs_layout`** ONNX metadata blob
> with every term's name, dim, scale, history and offset (plus control period,
> gait period, command dim, scandot geometry and the arm DOF indices). See
> [v2 layout metadata](#v2-layout-metadata). `evaluation/sim2sim.py` builds the
> observation from that blob and falls back to the table below only when the key
> is absent. Adding one term renumbers everything after it, so a hand-maintained
> table is exactly the thing that already cost this project a run.

For `a3_ultra_fast_sac` and `a3_ultra_loco_v2_s0` (history_length 1, 29 DOF →
**100 dims**):

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

### v2 layout metadata

`A3UltraFastSACAgent.export` (in `holosoma_ext/a3_ultra_loco_v2.py`) adds one more
key, `actor_obs_layout`:

```jsonc
{
  "groups": [                      // in actor_obs_keys order; concatenated in this order
    {"name": "actor_obs", "history_length": 5,
     "terms": [                    // ALREADY alphabetically sorted, with offsets
       {"name": "actions", "dim": 29, "scale": 1.0, "noise": 0.0, "start": 0, "width": 145},
       // ... base_ang_vel, command_ang_vel, command_lin_vel, cos_phase, dof_pos,
       //     dof_vel, heading_error, projected_gravity, sin_phase, upper_body_target
     ]},
    {"name": "perception_obs", "history_length": 1,
     "terms": [{"name": "height_scan", "dim": 117, "scale": 1.0, "start": 0, "width": 117}]}
  ],
  "total_dim": 692,
  "control_dt": 0.02,              // sim.fps / control_decimation — 100 Hz runs land here
  "clip_observations": 100.0,
  "gait_period": 1.0,
  "command_dim": 4,                // 4 once the heading command is on
  "scandots": {"nx": 13, "ny": 9, "spacing": 0.1, "x_offset": 0.2,
               "nominal_height": 1.05, "clip": 1.0},
  "arm_dof_indices": [15, ..., 28],
  "extension": "a3_ultra_loco_v2"
}
```

`start` is relative to its own group; a consumer walks the groups in order and
accumulates. Actual observed dims per stage: **s0 100, s1 575, s2 645, s3/s4 692**.

The three v2-only terms:

| term | dims | how a consumer produces it |
| --- | --- | --- |
| `heading_error` | 1 | `wrap_to_pi(target_yaw - yaw)`, or **0** when the episode is on a pure yaw-rate command (which is ~20% of training episodes, and all of v1's) |
| `upper_body_target` | 14 | the arm joint target about to be applied, minus the default pose. With no skill attached that is `action_scale * a_{t-1}` on the arm columns |
| `height_scan` | 117 | 13×9 downward grid in the base **yaw** frame, row-major over (x forward, y lateral), `clip(base_z - nominal_height - ground_z, ±clip)`. Reshapes to the `(1, 13, 9)` CNN encoder input |

The velocity estimator (component **B**) needs nothing from the consumer: its head
lives **inside** the exported actor, so it is already part of the graph.

## Consumers

1. **This repo's sim2sim gate** — `scripts/eval/sim2sim_suite.py`. MuJoCo classic,
   50 Hz, the generated training MJCF. See `docs/sim2sim_locomotion_report.md`.
2. **Holosoma's own inference stack** — `holosoma_inference/run_policy.py`
   reconstructs observations from the same metadata; it needs an inference-side
   RobotConfig for the A3 before it can be used.
