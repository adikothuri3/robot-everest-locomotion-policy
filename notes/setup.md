---
title: Machine & Environment
updated: 2026-08-13
status: current
---

# Machine & environment

Must reflect the machine's **current** state — update whenever something is installed or changed. Deep detail: `docs/environment_report.md` and `docs/tooling.md`.

## Hardware

Windows 11 Home, Ryzen 7 5700 (8C/16T), 16 GB RAM, **RTX 4060 Ti 8 GB**, driver 610.62 (CUDA 13.3), WSL2 Ubuntu-24.04 with GPU passthrough.

**8 GB VRAM consequences:** `num_envs` ≈ 512–2048, no camera-based training, Isaac Lab below its 16 GB min spec (headless only). Real training runs go to the cloud ([[decisions]]).

## Environments

| Environment | Contents |
| --- | --- |
| `.venv/` (Windows, py3.11) | mujoco 3.11.0, numpy, pytest, imageio+ffmpeg, this package editable |
| `.venv-isaac/` (Windows, py3.11) | Isaac Sim 5.1.0 (pip), Isaac Lab v2.3.0 editable, RSL-RL 3.0.1, torch 2.7.0+cu — the PhysX validation leg |
| WSL2 `~/holosoma/.venv/hsmujoco` (py3.12) | Holosoma @ `6e146b0`, **mujoco_warp git `ecaef88` (pinned — PyPI 3.11.0 breaks physics)**, warp-lang 1.16.0, torch 2.13.0+cu130 — MJWarp **smoke only** |
| cloud instance, conda `hssim` | Holosoma @ `6e146b0` + IsaacSim (`scripts/setup_isaacsim.sh`) — where the default `simulator:isaacsim` actually runs |

> [!warning] IsaacSim is the default backend but has no local home
> IsaacSim is not supported inside WSL2 on consumer setups, and this box is under Isaac Lab's 16 GB VRAM minimum. `scripts/train/train_a3_wsl.sh` therefore defaults to `isaacsim` and **fails with instructions** unless a `hssim` conda env exists; local runs need `SIMULATOR=mjwarp` (smoke) and real runs go to the cloud. See [[decisions]].

## Standing quirks

- **Norton TLS interception** breaks git/pip: `git config --global http.sslBackend schannel` and `pip config set global.use-feature truststore`. WSL pip needs the WSL-side equivalent.
- **Warp ≥1.16 removed `wp.types.array`** — re-apply `scripts/setup/patch_holosoma_warp.py` after any holosoma update.
- **WSL `nohup` jobs die** when the WSL session detaches — keep the session attached for long local runs (moot for training now that it's cloud-side).
- CRLF: scripts run in WSL must stay LF (`.gitattributes` handles it; check when adding new `.sh`).
- Windows/WSL split: Holosoma lives in WSL (`~/holosoma`); everything else (repo, assets, eval) on Windows, reached from WSL via `/mnt/c/...`.

## Smoke commands

```
make check-isaac      # Isaac Lab / PhysX headless + cross-sim comparison (primary sim leg)
make check-model      # MuJoCo A3 diagnostics (12 checks, independent-physics gate)
make test             # pytest (manifest/adapter/terrain)
make train-a3-smoke   # local MJWarp smoke run (SIMULATOR=mjwarp, 256 envs)
```
