# Everest locomotion — common commands.
# Windows: use `make` from Git Bash, or copy the commands directly.

PY := .venv/Scripts/python.exe
WSL := wsl -d Ubuntu-24.04 -- bash

.PHONY: check-model check-isaac convert-assets test train-a3 train-a3-smoke train-a3-ppo eval-stand eval-suite

check-model:        ## MuJoCo model diagnostics (12 PASS/FAIL checks)
	$(PY) scripts/diagnostics/check_mujoco_model.py

check-isaac:        ## Isaac Lab A3 validation (headless, needs .venv-isaac)
	.venv-isaac/Scripts/python.exe scripts/diagnostics/check_isaac_a3.py

convert-assets:     ## regenerate Holosoma 29-DOF A3 asset from official model
	$(PY) scripts/convert/make_holosoma_asset.py

test:               ## run the pytest suite
	$(PY) -m pytest -q

train-a3:           ## full A3 Ultra FastSAC training (WSL2/MJWarp)
	$(WSL) scripts/train/train_a3_wsl.sh fastsac --training.num_envs 1024

train-a3-smoke:     ## short A3 training smoke run
	$(WSL) scripts/train/train_a3_wsl.sh fastsac --training.num_envs 256 --algo.config.num_learning_iterations 300

train-a3-ppo:       ## A3 Ultra PPO arm (E07 comparison)
	$(WSL) scripts/train/train_a3_wsl.sh ppo --training.num_envs 1024

eval-stand:         ## stability suite on the PD-stand baseline (pipeline check)
	$(PY) scripts/eval/stability_suite.py --policy stand

eval-suite:         ## stability suite on an ONNX policy: make eval-suite ONNX=path/to.onnx
	$(PY) scripts/eval/stability_suite.py --policy onnx --onnx $(ONNX)
