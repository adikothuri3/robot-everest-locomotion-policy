# Configuration map

Two config systems coexist deliberately:

1. **`configs/robots/*.yaml`** — the canonical robot manifest (this repo's source of
   truth; consumed by the MuJoCo adapter, diagnostics, stability suite, tests).
2. **Holosoma typed-dataclass presets** — training-side configuration. This *is*
   Holosoma's structured config system (registry + CLI overrides, e.g.
   `--training.num_envs 1024`, `--randomization...`, `exp:a3-ultra-fast-sac`), and we
   extend it via `src/everest_locomotion/holosoma_ext/a3_ultra_presets.py` rather
   than duplicating a parallel Hydra tree that could drift. Every run dumps its full
   resolved config to `logs/<project>/<run>/holosoma_config.yaml` for reproducibility.

Where each concern lives:

| Concern | Location |
| --- | --- |
| joint order / limits / default pose / PD assumptions | `configs/robots/a3_ultra.yaml` |
| training algo + curriculum + DR + rewards | Holosoma presets (`a3_ultra_presets.py`) + CLI overrides |
| terrain interface for evaluation & future Everest terrain | `src/everest_locomotion/terrains/` (`TerrainSpec`) |
| evaluation scenarios | `scripts/eval/stability_suite.py` (CLI flags; results JSON records params) |
| experiment definitions & exact commands | `experiments/README.md` |

The `training/ terrain/ randomization/ evaluation/` subdirectories are reserved for
future YAML overlays (e.g., alpine curriculum stages) once the Isaac Lab arm needs
shared config; do not add configs there without wiring them to a consumer.
