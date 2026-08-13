# Cross-simulator robot consistency (Phase 7)

Goal: the policy must never be trained against a materially different robot than
the one it is evaluated on. Three simulators are in play:

| Simulator | Role | Asset used |
| --- | --- | --- |
| MJWarp (Holosoma, WSL2) | training | `assets/a3_ultra/holosoma/a3_ultra_29dof.xml` |
| MuJoCo classic (Windows) | validation / stability suite | same XML (identical by construction) |
| Isaac Sim/PhysX (Isaac Lab) | secondary training / cross-physics eval | `a3_ultra_29dof.urdf` (converted) |

## Automated comparison
```powershell
.venv\Scripts\python  scripts/diagnostics/dump_model_properties.py --sim mujoco
.venv-isaac\Scripts\python scripts/diagnostics/dump_model_properties.py --sim isaac
.venv\Scripts\python  scripts/diagnostics/compare_sim_properties.py results/consistency/mujoco.json results/consistency/isaac.json
```
Compares: total mass, per-body masses, joint limits, actuated-joint sets.
Tolerances: 0.01 kg/body, 0.002 rad, 0.05 kg total.

## Identity-by-construction guarantees
- MJWarp and MuJoCo classic consume the *same XML file* — geometry, inertia,
  limits, contacts are bit-identical; only the solver differs (Holosoma's CI
  additionally tests DR consistency across backends).
- The XML and URDF both derive from the official AgibotTech model at pinned
  commit `589f508` via `scripts/convert/make_holosoma_asset.py` (mechanical
  transforms only; verified mass preserved to 0.002 kg — the 2 g delta is the
  two added massless foot-contact-point bodies).

## Things that intentionally differ between engines (watch list)
- **PD control location**: Holosoma applies software PD (`control_type="P"`);
  Isaac Lab uses implicit solver PD (`ImplicitActuatorCfg`). Same gains ≠ same
  effective stiffness at low sim rates — validate with the settle test in both.
- **Contact model**: MuJoCo soft contacts (solref 0.005) vs PhysX iterative
  solver; foot contact = 13 tiny spheres/foot; PhysX may prefer a simplified
  collision (boxes) — if we simplify for PhysX, re-run the comparison and
  document the deviation here.
- **Joint passive damping/frictionloss**: baked into the MJCF (official values);
  the URDF `<dynamics>` values should match — the comparator checks limits and
  masses; damping comparison TODO once Isaac side runs.
- **Armature**: assumed values, applied via Holosoma config (MJWarp) and
  ArticulationCfg (Isaac); keep the single source in the manifest.

## Findings so far (2026-08-13)
1. **Caught: Isaac URDF import inflated the robot by 19.35 kg.** The official
   URDF gives 11 decorative/sensor links (torso shell, cameras, LiDAR, knee
   shells) mass=0 and inertia=0; Isaac's importer auto-computes mass from mesh
   volume — the torso shell alone became 18.6 kg. Fixed mechanically in
   `make_holosoma_asset.py` (tiny valid inertials stamped on zero-mass links);
   the comparator now verifies conservation via total mass.
2. **Topology fold difference (benign, warn-only):** the official MJCF folds
   fixed child links (IMU housings, hand palms) into parents; the URDF keeps
   them separate. Per-body masses differ where body sets differ; totals match.
3. **A3-specific MJWarp constraint:** the official per-link mesh collisions
   overflow MJWarp's per-env constraint allocation (`nefc overflow — njmax`).
   Training asset now uses 10 AABB-fitted primitive boxes + official foot
   spheres (verified: no self-penetration at default pose, mass preserved).

## Status
- MuJoCo-side dump complete (`results/consistency/mujoco.json`: 60.1796 kg, 29
  actuated). Isaac-side re-dump with the fixed URDF in progress via
  `scripts/diagnostics/check_isaac_a3.py`.
