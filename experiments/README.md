# Experiment ladder

All Holosoma commands run inside WSL2 (`wsl -d Ubuntu-24.04 -- bash ...`) with the
env activated (`source ~/holosoma/.venv/hsmujoco/bin/activate`) or via
`scripts/train/train_a3_wsl.sh`. `IMPORT=--import-file /mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py`.

Guideline for this 8 GB GPU: `--training.num_envs 512–1024` (G1 smoke used 512 at
~3.8 GB VRAM). Every run records: git commit (log it in the run notes), seed,
config dump (holosoma writes `holosoma_config.yaml` per run), TensorBoard events.

| ID | Purpose | Command core | Status |
| --- | --- | --- | --- |
| E00 | Upstream repro (G1 FastSAC, validates install) | `exp:g1-29dof-fast-sac simulator:mjwarp --training.num_envs 512 --algo.config.num_learning_iterations 300` | run 2026-08-13 (see bootstrap report) |
| E01 | A3 stand (pipeline sanity: zero-velocity commands dominate early training; also PD-stand floor via stability suite) | `exp:a3-ultra-fast-sac $IMPORT --training.num_envs 512 --algo.config.num_learning_iterations 200 --algo.config.logging_interval 10 --algo.config.save_interval 200` | run 2026-08-13 (`logs/everest-a3/20260813_233010`): GPU/WarpBackend, 0 nefc overflows, ~30 s/it wall-clock on this machine |
| E02 | A3 flat walking, faithful upstream recipe | `exp:a3-ultra-fast-sac $IMPORT --training.num_envs 1024` (full iterations) | ready |
| E03 | + moderate DR | same as E02 (upstream DR is already on; reduce/increase via `--randomization.*` overrides) | ready |
| E04 | + push disturbances | upstream pushes are on by default; sweep `--randomization` push magnitude | ready |
| E05 | rough terrain | terrain mix is default (`terrain_locomotion_mix`); increase difficulty via terrain overrides | ready |
| E06 | Everest stability reward | `exp:a3-ultra-fast-sac-everest $IMPORT` | ready (reward variant registered) |
| E07 | PPO vs FastSAC | `exp:a3-ultra-ppo $IMPORT` vs E02 (only after both reasonably tuned) | ready |
| E08 | Cross-physics eval | export ONNX (`eval_agent.py --training.export_onnx=True`) → `scripts/eval/stability_suite.py --policy onnx` (MuJoCo classic) → later Isaac | infra ready |
| E09 | strong terrain + strong DR | E05 + widened DR ranges | needs E05 |
| E10 | first alpine curriculum | Tomasz terrain via `TerrainPatch` → custom holosoma terrain term or mesh export | blocked on terrain generator handoff |
