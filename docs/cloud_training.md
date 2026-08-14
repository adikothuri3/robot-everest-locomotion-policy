# Cloud training — stable A3 Ultra walking

Local training works but is ~30 s/iteration on this machine (WSL2 + 4060 Ti);
the upstream FastSAC recipe was designed for a 24 GB-class Linux GPU where the
full 50k-iteration run takes on the order of **15-60 minutes**. Train in the
cloud, validate locally.

## What to rent
| Provider | Instance | Notes |
| --- | --- | --- |
| Lambda Cloud | 1× RTX 4090 or A10 | cheapest match to the paper's setup |
| RunPod / Vast.ai | RTX 4090 24 GB | choose Ubuntu 22.04/24.04 CUDA ≥ 12.4 image, driver ≥ 555 |
| AWS | g6e.xlarge (L40S) or g5.2xlarge (A10G) | Ubuntu 22.04 DLAMI fine |

Requirements: Ubuntu 22.04/24.04, NVIDIA driver ≥ 555.58.02, ≥ 24 GB VRAM
recommended (4096 envs; 8-16 GB works with `NUM_ENVS=1024-2048`), ~40 GB disk.

## Launch
```bash
# on the instance
git clone <this-repo-url> everest && cd everest
bash scripts/cloud/train_a3_cloud.sh
# options:
WANDB_API_KEY=xxxx EXP=a3-ultra-fast-sac NUM_ENVS=4096 ITERATIONS=50000 bash scripts/cloud/train_a3_cloud.sh
```
The script is fully pinned and self-verifying — see the header comments for
exactly which components are pinned and why (mujoco_warp PyPI 3.11.0 breaks
physics with holosoma 6e146b0; the script installs holosoma's own pinned
commit instead).

Note: the repo must include `assets/a3_ultra/holosoma/` (committed) — the
official AgibotTech model repo is NOT needed on the cloud box.

## Which simulator backend (important)
**`isaacsim` is the repo-wide default** — the `a3_ultra_*` presets, the WSL
training script, the `make train-a3*` targets, and this script all select it;
MJWarp must be requested explicitly. Holosoma's own nightly training
matrix validates FastSAC on `[isaacgym, isaacsim]` only; the MJWarp backend is
smoke-tested but not training-validated upstream, and we reproduced
deterministic physics NaNs under untrained-policy flailing on MJWarp with the
upstream G1 asset (pinned versions, 2026-08-13; A3 ruled out by control
experiment; best MJWarp combo found — mujoco 3.11.0 + mujoco_warp `ecaef88` +
warp-lang 1.15.0 — still NaNs by ~step 85).
Use `SIMULATOR=mjwarp` only for short smoke runs.

The IsaacSim backend imports the **URDF** (`usd_file=None` → dynamic URDF→USD
conversion) and then asserts that the articulation's body list equals the
preset's `body_names`. The generated URDF is built to satisfy exactly that —
34 links, contact-point links present, no decorative links (see
`scripts/convert/make_holosoma_asset.py` and `tests/test_asset_body_parity.py`,
which fails the build if the two assets ever drift). Pre-flight it locally with
`make check-isaac`, which imports the URDF with the same converter settings
Holosoma uses.

## Which experiment
1. **`a3-ultra-fast-sac` (default)** — upstream FastSAC sim-to-real recipe with
   the A3 swapped in: rough-terrain mix, push perturbations every 5-10 s, full
   dynamics randomization (friction, link mass, base mass ±, CoM shift, PD
   gains, torque RFI, action latency), action-rate curriculum. This is the
   hardware-validated robustness recipe — run it first.
2. **`a3-ultra-fast-sac-everest`** — same + stability-first reward additions
   (stumble penalty, foothold penalty, soft joint-limit penalty, alive 15).
   Run second; compare on the stability suite; keep the winner.
3. `a3-ultra-ppo` — PPO arm for the E07 comparison (25k iterations).

Suggested protocol for "very stable walking": run 1 and 2 with seeds {1,2,3},
score all six checkpoints with the stability suite, pick by survival + max
recoverable push, not by tracking error.

## Retrieve + validate locally
```powershell
scp -r <user>@<instance>:~/everest/checkpoints/cloud_* checkpoints/
# cross-physics validation (MuJoCo classic, NOT the training engine):
.venv\Scripts\python scripts/eval/stability_suite.py --policy onnx --onnx checkpoints/cloud_<run>/model_0050000.onnx
```
Also compare against the recorded floor: `results/stability/pd_stand_baseline.json`
(PD-stand survives 25/68 scenarios, 0.2-0.3 m/s pushes — a walking policy must
dominate this).

## Cost/wall-clock expectations
- FastSAC paper: full sim-to-real G1 policy in ~15 min on one RTX 4090
  (4096 envs). A3 is heavier (60 kg, 29 DOF vs 29 DOF) — same order.
- Budget one GPU-hour per (experiment × seed) including setup; the 6-run
  protocol above ≈ one afternoon on a single 4090.

## Known-good stack (pinned by the script)
| Component | Version |
| --- | --- |
| holosoma | `6e146b0` (2026-08-11) |
| mujoco_warp | git `ecaef88` (holosoma's pin; == 3.10.0) |
| mujoco | 3.11.0 (pulled by holosoma) |
| warp-lang | ≥1.14 (1.16 OK with `patch_holosoma_warp.py`, applied by script) |
