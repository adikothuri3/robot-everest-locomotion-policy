# Baseline Selection — A3 Ultra Extreme-Stability Locomotion

Decision date: 2026-08-13. Full research notes: `docs/research/`.

## Candidates surveyed

| | Holosoma (FastSAC/PPO) | Isaac Lab + RSL-RL PPO | AGIBOT_x1_train | Humanoid-Gym | Cross-embodiment (H-Zero, XHugWBC) | InstinctLab (Hiking in the Wild) |
| --- | --- | --- | --- | --- | --- | --- |
| Robots | Unitree G1, Booster T1 (29 DOF) | H1, G1 + any via URDF→USD | AgiBot X1 (12-DOF legs) | RobotEra XBot-S/L | 12-13 sim embodiments | Unitree G1 |
| Simulator | IsaacGym / IsaacSim 5.1 / **MJWarp**; MuJoCo for eval | Isaac Sim 4.5-5.1 (PhysX) | Isaac Gym Preview 4 (legacy) | Isaac Gym Preview 4 (legacy) | Isaac Gym (paper) | Isaac Lab 2.3.2 + InstinctMJ (MuJoCo) |
| Algorithm | FastSAC (distributional off-policy) + PPO | PPO (RSL-RL) | "DHPPO" (dual-history PPO + vel estimator) | PPO | PPO variants | PPO + depth encoder |
| Networks | MLP + obs-norm, auto action scaling | MLP [512,256,128] ELU | Actor[512,256,128]+CNN long-history encoder | MLP + 15-frame stack | MLP + morphology descriptors | CNN(depth)+MLP |
| Terrain | plane, rough, slopes, stairs, custom mesh + curriculum | ROUGH_TERRAINS_CFG + promote/demote curriculum | trimesh curriculum (rough/slope/discrete) | mostly flat (rough basic) | flat | extreme outdoor, foothold-aware |
| Perception | proprio (blind) | proprio + height-scan | proprio | proprio | proprio | **depth camera** |
| Domain rand. | friction, mass, CoM, PD, torque RFI, latency buffers, pushes | friction, mass, pose/vel resets, pushes | friction, mass, motor offset, PD, **5-40-step actuator lag**, pushes | simpler set | n/a | yes |
| Checkpoints | G1+T1 ONNX in-repo (morphology-locked) | none | none | 1 XBot JIT policy | **none public** | none found |
| HW validation | G1, T1 real deployments | H1/G1 lineage (ETH/legged) | X1 real | XBot-S/L real, zero-shot | none verifiable | full-size humanoid outdoors, 2.5 m/s |
| A3 port difficulty | moderate (extension system designed for it) | moderate (URDF→USD + config copy) | high (12-DOF template, legacy stack) | high (legacy stack) | impossible (no code) | moderate but perception-first |
| Major risks | Linux-only (→WSL2 here); 8 GB VRAM unverified | 16 GB VRAM min-spec (8 GB here); no built-in sim2sim/deploy | **no license**; frozen; Isaac Gym deprecated | dormant; Isaac Gym deprecated | vaporware until code drops | CC BY-NC; needs camera sim (won't fit 8 GB) |

## Scoring (1-5, weighted)

| Criterion | W | Holosoma | IsaacLab RSL-RL | X1 | Hum-Gym | Cross-emb | InstinctLab |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stability / disturbance robustness | 20% | 5 | 4 | 3 | 3 | 2 | 4 |
| Rough-terrain ability | 20% | 4 | 4 | 3 | 2 | 1 | 5 |
| A3 portability | 15% | 4 | 4 | 2 | 2 | 1 | 3 |
| Sim-to-real design | 10% | 5 | 3 | 4 | 4 | 1 | 4 |
| Pretrained weights | 10% | 3 | 2 | 1 | 2 | 1 | 1 |
| Isaac Lab compatibility | 10% | 4 | 5 | 1 | 1 | 1 | 5 |
| MuJoCo validation ability | 5% | 5 | 2 | 4 | 4 | 1 | 4 |
| Code quality | 5% | 5 | 4 | 3 | 3 | — | 4 |
| Active maintenance | 3% | 5 | 5 | 1 | 1 | — | 4 |
| License | 2% | 5 | 5 | 1 | 4 | — | 2 |
| **Weighted total** | | **4.35** | **3.75** | 2.50 | 2.42 | ~1.2 | 3.81 |

## Decision

### Primary baseline: **Holosoma + FastSAC** (IsaacSim/PhysX backend, cloud)

> **Backend superseded 2026-08-13.** This section chose MJWarp as the backend
> because it was the only stack that ran locally. It is not training-validated
> upstream and NaNs under untrained-policy flailing, so the default is now
> `simulator:isaacsim` (cloud), with MJWarp kept for local smoke. The *trainer*
> choice (Holosoma + FastSAC) is unchanged. See `notes/decisions.md`.

The hypothesis held. Decisive factors beyond the score:
- The FastSAC recipe *is* the stability recipe we want: trained **with** rough terrain, pushes, heavy DR and an action-rate curriculum, and validated on two real humanoids.
- MJWarp backend runs on this machine's WSL2 (Ubuntu 24.04 + Python 3.12 is an officially supported combo; driver OK) — the only modern GPU-parallel training stack that does.
- Native MuJoCo sim-to-sim evaluation closes our cross-physics loop cheaply (train MJWarp → validate MuJoCo + later Isaac).
- Apache-2.0, active, extension system designed for adding robots without forking.
- Risk accepted: 8 GB VRAM is untested upstream → mitigate with reduced `num_envs`/replay size; E00 (reproduce G1 upstream) validates the install before any A3 work.

### Fallback / conservative baseline: **Isaac Lab 2.3.x + RSL-RL PPO rough-terrain (G1/H1 config pattern)**
- Native Windows pip install; the project's broader work already targets Isaac Lab; terrain-curriculum machinery aligns with Tomasz's terrain-generator integration.
- Below min-spec VRAM is the main risk (headless, ~1-2k envs expected workable).
- Serves as the E07 PPO comparison arm and the PhysX side of cross-sim validation.

### Reference benchmark: **AGIBOT_x1_train conventions + Humanoid-Gym sim2sim methodology**
- Not runnable here (legacy Isaac Gym, Linux-only, X1 has no license file ⇒ no code reuse).
- What we take (as *documented ideas*, re-implemented): 100 Hz control precedent, actuator/observation lag randomization ranges (5-40 steps), dual-history + velocity-estimator architecture idea, Isaac→MuJoCo sim2sim acceptance-test discipline.

### Earmarked for Phase 19 (perception): **InstinctLab**
Highest terrain score of all candidates, and it is an Isaac Lab extension (drop-in ArticulationCfg port). Blocked for now by: perception-first design (violates our "no perception dependency" rule), camera sim VRAM, CC BY-NC 4.0 license (fine for research; flag before any product use).

### Explicitly rejected
- **Cross-embodiment pretrained policies**: no public code or weights exist as of Aug 2026 (H-Zero paper-only; XHugWBC "coming soon"). Re-evaluate if XHugWBC releases.
- **Checkpoint reuse from G1/T1**: obs/action dims and joint order are morphology-locked (A3: 29 controlled DOF at different limits/masses). Retraining from scratch with FastSAC is cheap (~minutes-hours), cheaper than any transfer surgery. See Phase 15 notes in `docs/bootstrap_report.md`.
