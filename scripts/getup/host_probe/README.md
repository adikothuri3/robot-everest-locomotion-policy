# HoST feasibility probe — A3 Ultra get-up (throwaway)

**Question this answers:** can HoST's recipe discover *any* get-up strategy at
60 kg with 24 Nm elbows? Nobody has demonstrated learned get-up above 35 kg
(G1). One week, one cloud GPU, code is throwaway — the durable implementation
is the Holosoma extension (see the approved plan / `docs/research/getup_recipes.md`).

Success = supine (then prone) rise in sim, any quality. Bring back:
1. TensorBoard reward curves per reward group (golden reference for the reimplementation)
2. Policy rollout videos of the discovered strategy
3. Per-joint torque traces during rise — which joints saturate (expect elbows/shoulders)

## Setup (Linux cloud GPU; Isaac Gym needs Python 3.8 + older CUDA userland)

```bash
conda create -n host python=3.8 -y && conda activate host
# Isaac Gym Preview 4: download from https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .
git clone https://github.com/InternRobotics/HoST && cd HoST
pip install -e legged_gym -e rsl_rl
```

Known landmines (from HoST issues, all unanswered by maintainers):
- #33: coredump on RTX 4090 — if hit, try `--sim_device cuda:0 --pipeline gpu`,
  reduce `num_envs`, or use an A100/L40S instance instead.
- #42: verify `smoothness_lower_bound > 0` in the PPO config actually reaches
  the L2C2 loss (grep `smoothness_lower_bound` in rsl_rl; if any code path
  forces 0, L2C2 silently no-ops — it's load-bearing for smoothness).
- #37: reward-function bug report — read the issue before trusting
  `style_ground_parallel`.
- #35: the one prior big-robot port (H1-2) "stood up but behaved strangely" —
  motion quality, not success rate, is the known failure mode at scale.

## Port steps

1. `mkdir -p legged_gym/resources/robots/a3_ultra` and copy from this repo:
   `assets/a3_ultra/holosoma/a3_ultra_29dof.urdf` + `assets/a3_ultra/holosoma/meshes/`.
   (URDF has head welded, zero-mass links fixed. Isaac Gym convex-hulls the
   collision meshes; if import is slow or contacts explode, strip `<collision>`
   meshes from distal links, mirroring our MJCF primitive boxes.)
2. `mkdir legged_gym/legged_gym/envs/a3_ultra` and copy
   `a3_ultra_config_ground.py` there; add an `a3_ultra_env.py` that subclasses
   the G1 env class unchanged (HoST's env logic is joint-name-driven via the
   cfg lists; start with zero env-code changes).
3. Register in `legged_gym/legged_gym/envs/__init__.py` mirroring the
   `g1_ground` lines: task `a3_ultra_ground` -> (A3 env class, A3UltraCfg, A3UltraCfgPPO).
   3b. If the env hard-requires the `keyframe`/`keyframe_head` marker bodies
   (grep `keyframe_head` in the G1 URDF/env), append fixed massless links with
   those names to the A3 URDF: parent `head_pitch_Link`, `keyframe_head` at the
   head top (~+0.15 m z from head_pitch origin; A3 head top ≈ 1.74 m standing).
4. First run, deliberately easy — confirm the pipeline before judging the recipe:
   ```bash
   cd legged_gym && python legged_gym/scripts/train.py --task a3_ultra_ground \
       --num_envs 4096 --headless --max_iterations 12000
   ```
5. Verify obs dim: our config guesses `num_one_step_observations = 94`
   (HoST G1: 76 = 7 + 3x23 -> 7 + 3x29 = 94). If train.py asserts on obs size,
   read `compute_observations()` and fix the constant.
6. Play/record: `python legged_gym/scripts/play.py --task a3_ultra_ground`.

## Tuning ladder if it doesn't rise (in order, change one thing per run)

1. Raise `curriculum.force` 175 -> 200–250 per trunk link (more training wheels;
   HoST anneals it away automatically once head height > threshold).
2. Slow the β anneal / raise its floor (grep `action_rescale`; G1 floor 0.25).
3. Relax `regu_*` weights x0.5 (we already cut torque/power penalties ~4x for
   the bigger motors; discovery needs freedom, smoothness comes later).
4. Raise arm stiffness 60 -> 100 (G1 value) — weak-armed PD may be the blocker,
   and sim kp is NOT deployment kp (HoST deploys at 1.33–1.5x sim kp anyway).
5. If elbows saturate in traces: accept forearm-support strategies — check
   `penalize_contacts_on` isn't punishing the only viable support path
   (drop `"elbow"` from the list for a diagnostic run).
6. If nothing rises in ~6 runs: that's a real answer. The recipe needs
   restructuring at this mass (leg-dominant style rewards, kneel-first staging)
   — take that straight into the Holosoma implementation instead of polishing
   the probe.

## Interpreting the result

- **Rises, roughly:** recipe transfers; proceed with Holosoma reimplementation
  per plan; keep curves/videos as the parity reference.
- **Rises only with pull-force floor > 0:** morphology is marginal — plan for
  kneel-first curriculum + possibly HumanUP Stage-II slow-down from the start.
- **Never rises:** the honest failure the research warned about. Pivot the
  Holosoma env toward staged leg-dominant strategies (side-roll -> kneel ->
  squat, arms as outriggers only) rather than free discovery.
