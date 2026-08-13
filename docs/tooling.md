# Agent / developer tooling

Recorded 2026-08-13.

## Tooling available in this development environment
- **Claude Code** agent environment with: file tools, PowerShell + Git Bash, WSL2 access, web search/fetch, DeepWiki MCP (AI Q&A over GitHub repos — used for Holosoma/IsaacLab research), Context7 MCP (library docs), browser automation (unused so far).
- No external "Isaac"/"MuJoCo" agent plugins were installed: none exist that are trustworthy/maintained; project-local reference material in `docs/research/` serves that role instead (per the operating principle of avoiding abandoned agent extensions).

## Project tooling installed
| Tool | Where | Purpose |
| --- | --- | --- |
| Python 3.11 venv `.venv/` | Windows | MuJoCo validation env: mujoco 3.11.0, numpy 2.4.6, pyyaml, pytest 9, imageio(+ffmpeg) |
| Python 3.11 venv `.venv-isaac/` | Windows | Isaac Sim 5.1.0 pip + Isaac Lab (conservative baseline) |
| uv venv `~/holosoma/.venv/hsmujoco` | WSL2 Ubuntu-24.04 | Holosoma training env: Python 3.12, torch 2.13.0+cu130, mujoco 3.11.0, mujoco-warp 3.11.0, warp-lang 1.16.0 |
| git (schannel SSL) | Windows | Norton TLS interception workaround |
| pip truststore feature | Windows (global pip.ini) | same workaround for pip |
| TensorBoard (via holosoma) | WSL2 | experiment tracking (wandb optional, no credentials configured) |

## Local patches to third-party code
| File | Why | Patch |
| --- | --- | --- |
| `~/holosoma/src/holosoma/holosoma/utils/warp_utils.py` | Warp 1.16 removed `wp.types.array(ptr=...)` | replaced with `wp.from_torch(x.contiguous(), dtype=wp.vec3)` (3 sites); script: `scripts/setup/patch_holosoma_warp.py` (idempotent). Consider upstreaming. |

## Key conventions for agents working in this repo
- Canonical joint order lives in `configs/robots/a3_ultra.yaml`; NEVER hand-copy joint lists — tests in `tests/test_manifest.py` cross-check the Holosoma preset against the manifest.
- Holosoma runs ONLY in WSL2 (`wsl -d Ubuntu-24.04`); Windows-side code must not import holosoma.
- Shell scripts written on Windows have CRLF; strip with `tr -d '\r'` before running in WSL (see existing scripts).
