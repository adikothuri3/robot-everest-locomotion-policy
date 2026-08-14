# Everest locomotion — common commands.
# Windows: use `make` from Git Bash, or copy the commands directly.

PY := .venv/Scripts/python.exe
WSL := wsl -d Ubuntu-24.04 -- env

.PHONY: check-isaac check-model convert-assets test train-a3 train-a3-smoke train-a3-ppo eval-stand eval-suite

check-isaac:        ## Isaac Lab / PhysX A3 validation — primary sim leg (needs .venv-isaac)
	.venv-isaac/Scripts/python.exe scripts/diagnostics/check_isaac_a3.py

check-model:        ## MuJoCo model diagnostics — cross-physics check (12 PASS/FAIL)
	$(PY) scripts/diagnostics/check_mujoco_model.py

convert-assets:     ## regenerate Holosoma 29-DOF A3 asset from official model
	$(PY) scripts/convert/make_holosoma_asset.py

test:               ## run the pytest suite
	$(PY) -m pytest -q

train-a3:           ## full A3 Ultra FastSAC training (IsaacSim; real runs go to the cloud script)
	$(WSL) bash scripts/train/train_a3_wsl.sh fastsac --training.num_envs 1024

train-a3-smoke:     ## short local smoke run (MJWarp — smoke-only backend)
	$(WSL) SIMULATOR=mjwarp bash scripts/train/train_a3_wsl.sh fastsac --training.num_envs 256 --algo.config.num_learning_iterations 300

train-a3-ppo:       ## A3 Ultra PPO arm (E07 comparison, IsaacSim)
	$(WSL) bash scripts/train/train_a3_wsl.sh ppo --training.num_envs 1024

eval-stand:         ## stability suite on the PD-stand baseline (pipeline check)
	$(PY) scripts/eval/stability_suite.py --policy stand

eval-suite:         ## stability suite on an ONNX policy: make eval-suite ONNX=path/to.onnx
	$(PY) scripts/eval/stability_suite.py --policy onnx --onnx $(ONNX)
