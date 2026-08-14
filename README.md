# Everest Locomotion — AgiBot A3 Ultra

RL locomotion research repository for the **AgiBot A3 Ultra** humanoid, targeting
**extreme stability**: fall resistance, disturbance recovery, slip resistance, and
(eventually) alpine/Everest-class rough terrain.

Research question: *what existing RL locomotion foundation gives the A3 Ultra the
strongest starting point for extremely robust locomotion, and how far can we
specialize it toward alpine terrain without sacrificing balance and recovery?*

## Current baseline (see `docs/baseline_selection.md`)
- **Primary: [Holosoma](https://github.com/amazon-far/holosoma) + FastSAC on the IsaacSim (PhysX) backend** — the default simulator for every preset and script. Rough terrain + pushes + heavy domain randomization, hardware-validated recipe, Apache-2.0.
- **Isaac Lab 2.3 + RSL-RL PPO** (native Windows, `.venv-isaac`) — same PhysX physics; validation leg and trainer fallback.
- **MuJoCo / MJWarp: validation and smoke only** — MuJoCo classic runs the stability suite and sim2sim gate; MJWarp is short local smoke runs only (NaNs under untrained flailing). See `notes/decisions.md`.
- **References: AGIBOT X1 stack conventions, Humanoid-Gym sim2sim methodology** (docs only — legacy Isaac Gym, not runnable here).

## Quickstart

### 0. Prerequisites (this machine is already set up — see `docs/environment_report.md`)
- Windows 11 + WSL2 Ubuntu-24.04 with NVIDIA driver ≥ 555 (GPU visible in WSL).
- Python 3.11 (Windows), git.

### 1. Clone third-party assets & create envs
```powershell
git config --global http.sslBackend schannel          # if TLS interception (Norton)
pip config set global.use-feature truststore          # same, for pip
git clone https://github.com/AgibotTech/A3-A3U-robot-model third_party/A3-A3U-robot-model
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

### 2. Validate the robot model in MuJoCo (12 PASS/FAIL checks + video)
```powershell
.venv\Scripts\python scripts/diagnostics/check_mujoco_model.py --render
.venv\Scripts\python -m pytest -q          # manifest/adapter/terrain tests
```

### 3. Set up Holosoma (WSL2 = the MJWarp smoke env) and reproduce upstream (E00)
```bash
# inside WSL2 Ubuntu-24.04
git clone https://github.com/amazon-far/holosoma ~/holosoma
cd ~/holosoma && bash scripts/setup_mujoco_via_uv.sh --no-robot-sdks
python /mnt/c/.../scripts/setup/patch_holosoma_warp.py ~/holosoma/src/holosoma/holosoma/utils/warp_utils.py
source .venv/hsmujoco/bin/activate
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:mjwarp \
  --training.num_envs 512 --algo.config.num_learning_iterations 300
```

### 4. Train the A3 Ultra
**Cloud (recommended — local wall-clock is ~30 s/iter):** on a Linux GPU box,
`bash scripts/cloud/train_a3_cloud.sh` — fully pinned and self-verifying; see
`docs/cloud_training.md`.
The default backend everywhere is `simulator:isaacsim`; MJWarp must be asked for
explicitly.
```powershell
# local smoke runs (WSL2, MJWarp — IsaacSim is not supported inside WSL2).
# NOTE: install the pinned mujoco_warp first (scripts/setup/wsl_pin_mjwarp.sh)
# — PyPI mujoco-warp 3.11.0 breaks physics
wsl -d Ubuntu-24.04 -- env SIMULATOR=mjwarp bash scripts/train/train_a3_wsl.sh fastsac --training.num_envs 512
# tasks: fastsac | ppo | everest | getup | getup-fast-sac
```
**Get-up** trains from the same stack: `bash scripts/cloud/train_a3_getup_cloud.sh`
on the cloud box (`make train-getup-smoke` is the local wiring check only).

### 5. Evaluate a trained policy (MuJoCo sim2sim gate)
```powershell
$P = "checkpoints/<run>/model_0050000.onnx"
.venv\Scripts\python scripts/eval/sim2sim_suite.py --mode sweep --run-dir checkpoints/<run>
.venv\Scripts\python scripts/eval/sim2sim_suite.py --mode showcase --onnx $P --name showcase --video
.venv\Scripts\python scripts/eval/sim2sim_suite.py --mode grid --onnx $P --name showcase --with-baseline
.venv\Scripts\python scripts/eval/stability_suite.py --policy stand --quick   # PD-stand floor only
```
Videos land in `results/videos/<name>/`, metrics in `results/sim2sim/`. The first
run's results: `docs/sim2sim_locomotion_report.md` (68/68 vs a 26/68 PD-stand
control). `stability_suite.py` records the floor and nothing else — trained
policies need the real observation contract, which only the sim2sim harness
implements (`docs/onnx_policy_interface.md`).

## Architecture

```
configs/robots/a3_ultra.yaml     canonical robot manifest (SINGLE SOURCE OF TRUTH
                                 for joint order/limits; official vs assumed values marked)
src/everest_locomotion/
  robots/manifest.py             manifest loader
  sim_adapters/mujoco_adapter.py MuJoCo mapping (canonical order <-> qpos/actuators) + PD
  sim_adapters/mujoco_scene.py   inject generated terrain heightfields into scenes
  terrains/                      TerrainSpec/TerrainPatch interface + procedural_rough
                                 (Tomasz's Everest generator plugs in here later)
  policies/                      Policy protocol; PDStand floor baseline
  evaluation/rollout.py          metric-instrumented rollouts (slip, tilt, saturation...)
  evaluation/sim2sim.py          exported-policy runner: holosoma's exact obs/action
                                 contract in MuJoCo classic (the trained-policy gate)
  holosoma_ext/a3_ultra_presets.py  Holosoma --import-file: A3 robot + locomotion experiments
  holosoma_ext/a3_ultra_getup.py    Holosoma --import-file: get-up task (env, rewards,
                                 pose-bank resets, assist-force curriculum)
assets/a3_ultra/holosoma/        generated 29-DOF training asset (MJCF + URDF, head
                                 welded, foot contact points) — regenerate via
                                 scripts/convert/make_holosoma_asset.py
assets/a3_ultra/getup/           fallen-pose bank (scripts/getup/generate_fallen_poses.py)
scripts/cloud/                   turnkey Lambda/cloud runs: locomotion + get-up
scripts/diagnostics/             model checks, cross-sim property dump/compare
docs/                            environment, research, baseline decision, reports
```

The two generated assets must describe the same robot — IsaacSim imports the
URDF, MuJoCo reads the MJCF, and Holosoma asserts the body list matches the
preset. `tests/test_asset_body_parity.py` fails if they drift.

**Joint-order rule:** the manifest order is legs (L,R) → waist → arms (L,R) → head.
The MJCF *kinematic tree* order differs — never index qpos directly; go through
`MujocoRobot`. Tests fail loudly on drift (`tests/test_manifest.py`).

## The A3 Ultra (T2.5 locomotion variant)
31 actuated DOF (12 leg + 3 waist + 14 arm + 2 head; head frozen for locomotion → 29
controlled). 60.18 kg, pelvis root, foot soles ~1.06 m below pelvis at the crouched
default pose. Official MJCF/URDF from AgibotTech (Mulan PSL v2), pinned in
`third_party/README.md`. G1-compatible joint naming (verified) makes Holosoma's G1
presets directly reusable.

## Training ladder (planned experiments)
Locomotion: E00 upstream repro → E01 stand → E02 flat walk → E03 +DR → E04 +pushes
→ E05 rough terrain → E06 Everest stability reward (`a3-ultra-fast-sac-everest`) →
E07 PPO vs FastSAC → E08 cross-physics eval → E09 strong terrain+DR → E10 alpine
curriculum.
Get-up: E11 HoST feasibility probe → E12 get-up task in Holosoma
(`a3-ultra-getup`) → E13 flat get-up cloud run → E14 chained get-up→locomotion
handoff → E15 slope/rough get-up. Definitions and status: `experiments/README.md`.

## Sim-to-sim validation
Train in IsaacSim/PhysX → evaluate in MuJoCo classic (`scripts/eval/sim2sim_suite.py`),
the independent-physics gate → Isaac Lab for PhysX-side articulation checks
(`make check-isaac`). Model-property consistency: `scripts/diagnostics/dump_model_properties.py`
+ `compare_sim_properties.py` (see `docs/simulator_consistency.md`).

**Status (2026-08-14):** the first cloud-trained locomotion policy passes the gate —
68/68 stability scenarios against a 26/68 PD-stand control on the same harness, and
2.5–4.0 m/s recoverable pushes against a 0.2–0.3 m/s floor. Full report and videos:
`docs/sim2sim_locomotion_report.md`.

## Known limitations
- 8 GB VRAM: keep `num_envs` ≈ 512–2048; no camera-based training on this machine.
- Isaac Lab: below NVIDIA's 16 GB minimum spec — headless-only, reduced envs.
- Windows/WSL split: Holosoma lives in WSL (`~/holosoma`), everything else on Windows.
- PD gains, armature, and default pose are **assumptions**, not AgiBot specs (marked in the manifest).
- Passive PD standing is statically unstable at RL gains (physics, not a bug):
  diagnostics use a documented stiff "hold mode".
