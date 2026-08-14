---
title: Open Questions & Risks
updated: 2026-08-13
status: current
---

# Open questions & risks

Things we don't know yet. Move an item to [[decisions]] (or delete it) once resolved.

- **Can a 60 kg humanoid learn to get up at all?** Nobody has shown learned get-up above ~35 kg (G1). A3 arms are weak for its mass (60/24/6 Nm shoulder/elbow/wrist) → favor leg-dominant strategies; waist pitch range disfavors sit-up routes. E11 (HoST Isaac Gym probe) exists to answer this cheaply before we invest in the Holosoma reimplementation.
- **PD gains, armature, and default pose are assumptions**, not AgiBot specs (marked in the manifest). Must be replaced with real actuator specs (AgiBot docs or A3 SDK) before any sim-to-real claim. Single edit point: `configs/robots/a3_ultra.yaml`.
- **Get-up reward weights are first-guess, untuned.** The extension's staged weights (upright 1 / height 2 / target-pose 2.5 / stand-still 1.5) and assist-curriculum thresholds (0.3/0.7, ±0.05 per 4096 resets) are HoST-inspired but never trained to convergence. Expect iteration on the first Lambda runs; watch `getup_success_rate` vs `getup_assist_scale` coupling (assist must reach 0 or the policy is a lie).
- **Assist-force IsaacSim branch source-verified but never executed.** Pre-cloud review confirmed against IsaacLab 2.3.0 source: call signature valid, wrench flushed by `scene.write_data_to_sim()` in the same substep, persistence semantics correct, body_ids/quat conventions correct. Two runtime caveats for the first cloud run: (a) `set_external_force_and_torque` is deprecated in 2.3 → expect DeprecationWarning spam (harmless; holosoma's own gantry/push hit the same path); (b) sanity-check assist actually changes behavior (a few iterations with `EVEREST_GETUP_DISABLE_ASSIST=1` as control). Also watch the first ~50 iterations for base_velocity termination spikes — bank poses were settled in MuJoCo, PhysX geometry differs (mitigated by +1 cm spawn clearance).
- **Holosoma MuJoCo-backend dof-order bug (found in get-up review, affects locomotion smoke too).** The MuJoCo/MJWarp wrapper orders dof arrays in MJCF TREE order (A3: waist-first) but fills `dof_vel_limits`/`torque_limits`/`dof_pos_limits` POSITIONALLY from the config lists (canonical legs-first) → scrambled per-joint limits on that backend only (G1 unaffected: its tree order == config order; IsaacSim unaffected: resolves by name with preserve_order). Local consequences: action-term torque clipping and `limits_dof_pos` reward use scrambled limits in ALL local MJWarp smoke runs (incl. locomotion E06 variant). The get-up extension now builds its own name-mapped limit tensors and is immune on every backend. Consider reporting upstream to amazon-far/holosoma. Cloud IsaacSim runs are unaffected.
- **Early reward is penalty-dominated until first rises.** At the penalty floor (scale 0.1) an untrained policy pays ~0.074/step action-rate against ~0.01–0.05/step task reward; the assist force creating height reward is what breaks the "lie still" optimum. Watch `rew_penalty_action_rate` vs task terms in the first 500 iterations; if the policy goes limp, drop PenaltyCurriculum `initial_scale`/`min_scale` to 0.05.
- **Does the v5 asset change locomotion behavior?** v5 adds 16 collision boxes over the E01 asset and makes hip links contact-capable — "hip" termination fires slightly earlier on falls. Watch nefc counters on the first v5 locomotion run.
- **Terrain handoff not delivered.** Tomasz's Everest terrain generator plugs into `TerrainSpec`/`TerrainPatch`; E10/E15 (the whole Everest fine-tune phase) blocked until it lands. Interim option: GeologicDome's existing reconstructions.
- **ONNX action scaling** — per-joint action scaling must be fixed per `docs/onnx_policy_interface.md` before cross-physics validation of a trained checkpoint (E08).
- **W&B credentials not configured** — TensorBoard only for now.
- **FastSAC vs PPO on A3** (E07) — undecided until both are reasonably tuned.
