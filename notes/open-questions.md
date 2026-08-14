---
title: Open Questions & Risks
updated: 2026-08-13
status: current
---

# Open questions & risks

Things we don't know yet. Move an item to [[decisions]] (or delete it) once resolved.

- **Can a 60 kg humanoid learn to get up at all?** Nobody has shown learned get-up above ~35 kg (G1). A3 arms are weak for its mass (60/24/6 Nm shoulder/elbow/wrist) → favor leg-dominant strategies; waist pitch range disfavors sit-up routes. E11 (HoST Isaac Gym probe) exists to answer this cheaply before we invest in the Holosoma reimplementation.
- **PD gains, armature, and default pose are assumptions**, not AgiBot specs (marked in the manifest). Must be replaced with real actuator specs (AgiBot docs or A3 SDK) before any sim-to-real claim. Single edit point: `configs/robots/a3_ultra.yaml`.
- **Does the v4 asset change locomotion behavior?** v4 adds 16 collision boxes over the E01 asset and makes hip links contact-capable — "hip" termination fires slightly earlier on falls. Watch nefc counters on the first v4 locomotion run.
- **Terrain handoff not delivered.** Tomasz's Everest terrain generator plugs into `TerrainSpec`/`TerrainPatch`; E10/E15 (the whole Everest fine-tune phase) blocked until it lands. Interim option: GeologicDome's existing reconstructions.
- **ONNX action scaling** — per-joint action scaling must be fixed per `docs/onnx_policy_interface.md` before cross-physics validation of a trained checkpoint (E08).
- **W&B credentials not configured** — TensorBoard only for now.
- **FastSAC vs PPO on A3** (E07) — undecided until both are reasonably tuned.
