# Environment Report

Inspected 2026-08-13.

## Hardware
| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5700 (8C/16T) |
| RAM | 16 GB (15.9 GiB usable) |
| GPU | NVIDIA GeForce RTX 4060 Ti, **8 GB VRAM**, WDDM mode |
| Disk | C: 874 GB free / 1.86 TB |

## Host software (Windows 11 Home 10.0.26200)
| Item | Value |
| --- | --- |
| NVIDIA driver | 610.62 (CUDA UMD 13.3) |
| CUDA toolkit | 13.3 (`nvcc` at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`) |
| Python | 3.9 / 3.10 / 3.11 / 3.12 via `py` launcher (system 3.12 has torch 2.6.0+cu124) |
| Git | 2.55.0.windows.3 |
| Conda / mamba / docker / uv | **not installed** |
| WSL2 | Ubuntu-24.04 (default), GPU passthrough verified (`nvidia-smi` works, driver 610.43.02), Python 3.12.3, ~7 GB RAM cap (default .wslconfig) |
| Isaac Sim / Isaac Lab | not installed (only an empty Omniverse cache dir) |
| MuJoCo / JAX | not previously installed |

## Network quirk
TLS interception (Norton) breaks default OpenSSL certificate verification:
- git: fixed globally via `git config --global http.sslBackend schannel`.
- pip: fixed via `pip config set global.use-feature truststore` (written to `%APPDATA%\pip\pip.ini`).
- WSL2: network works normally (no interception observed on first test).

## Installation strategy chosen
1. **Project venv (Windows, Python 3.11)** at `.venv/` — MuJoCo 3.11.0, numpy 2.4.6, pyyaml, pytest, imageio. Python 3.11 chosen for forward-compatibility with Isaac Sim 5.1 pip packages (which require 3.11). System Python untouched.
2. **Holosoma (primary baseline)** runs Linux-only → installed inside **WSL2 Ubuntu-24.04** using its `setup_mujoco_via_uv.sh` (MJWarp training backend; Ubuntu 24.04/Python 3.12 is an officially supported combination; driver 610 ≥ required 555.58.02).
3. **Isaac Lab (conservative baseline)**: native Windows pip install (`isaacsim[all,extscache]==5.1.0` + IsaacLab 2.3.x) in a separate 3.11 venv. **Risk: official minimum is 16 GB VRAM; we have 8 GB.** Headless blind/height-scan locomotion with reduced `num_envs` is expected to work (community-reported), tiled-camera rendering will not.

## Constraints to remember
- 8 GB VRAM: cap parallel envs (thousands, not tens of thousands); no camera-based training.
- 16 GB system RAM: WSL2 defaults to ~7 GB; raise via `%UserProfile%\.wslconfig` (`memory=12GB`) if Holosoma replay buffers OOM.
- Windows 11 **Home**: no Hyper-V manager, but WSL2 works; Docker Desktop not installed (not required by chosen strategy).
- WDDM GPU mode (not TCC): fine for all chosen stacks.
