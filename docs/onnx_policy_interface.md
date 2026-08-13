# Exported policy interface (Holosoma FastSAC/PPO → ONNX)

For E08 cross-physics evaluation there are two consumers of an exported A3 policy:

## 1. Holosoma's own inference stack (primary, exact)
`holosoma_inference/run_policy.py` reconstructs observations from the ONNX
metadata + robot config and is the reference sim2sim path (MuJoCo classic).
Requires an inference-side RobotConfig for the A3 (mirror of the training
preset; add to `holosoma_ext` when the first checkpoint exists).

## 2. This repo's stability suite (metrics harness)
`scripts/eval/stability_suite.py --policy onnx --onnx <file> --obs-layout ...`
The actor observation for `g1_29dof_loco_single_wolinvel` (reused by the A3
experiments) is a single concatenated vector (history_length=1), term order:

| # | term | dims | scale | note |
| --- | --- | --- | --- | --- |
| 1 | base_ang_vel | 3 | 0.25 | body frame |
| 2 | projected_gravity | 3 | 1.0 | |
| 3 | command_lin_vel | 2 | 1.0 | vx, vy |
| 4 | command_ang_vel | 1 | 1.0 | yaw rate |
| 5 | dof_pos | 29 | 1.0 | relative to default pose, canonical order |
| 6 | dof_vel | 29 | 0.05 | |
| 7 | actions | 29 | 1.0 | previous raw policy output |
| 8 | sin_phase / cos_phase | 2 | 1.0 | gait clock |
| ... | (verify against the run's `holosoma_config.yaml` — total actor dim printed at setup, e.g. 100) | | | |

Rules:
- FastSAC uses an **observation normalizer**; the exported ONNX embeds it — feed
  RAW scaled observations as above, do not re-normalize.
- Action output is in [-1, 1]; Holosoma rescales by per-joint action bounds
  (printed at setup as "Scaling: tensor([...])") around the default pose. The
  suite's `OnnxPolicy` must apply the same per-joint scaling — extend it before
  first real use (current implementation applies a single scalar `action_scale`,
  which is only correct for uniform bounds).
- Always cross-check the exact term list against the training run's
  `holosoma_config.yaml` (saved per run) rather than trusting this table.
