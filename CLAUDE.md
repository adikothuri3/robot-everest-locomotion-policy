# robot-everest-locomotion-policy — A3 Ultra locomotion + get-up

**Objective:** a smooth, stable, working **locomotion policy + get-up policy** for the **AgiBot A3 Ultra** (the only robot in scope), which then get **fine-tuned on Everest-like terrain** — ending with a stable and smooth alpine locomotion + get-up policy.

Stack: Holosoma + FastSAC on **IsaacSim (PhysX) — the default simulator everywhere**; Isaac Lab 2.3 + RSL-RL as the PhysX validation leg and trainer fallback; MuJoCo/MJWarp for sim2sim evaluation and local smoke only. Training in the cloud, validation local. Baselines, floors-to-beat, and scope boundaries: `notes/baselines.md`.

Everest terrain comes from the sibling repo `C:\Users\Aditya\VSCode\GeologicDome` (footage → LingBot-Map → sim terrain; its own vault is in `GeologicDome/notes/`). This repo consumes that terrain via `TerrainSpec`/`TerrainPatch`; it does not rebuild that pipeline. This repo's vault conventions mirror GeologicDome's.

## Start here

**Read `notes/overview.md` first on every cold start.** Before any nontrivial task, read all of `notes/` — six short files, flat, each readable in under 5 minutes.

## The vault (`notes/`)

`notes/` is an Obsidian vault whose only purpose is giving an agent (or Aditya) enough context to work on this project. It is not a journal and not a PKM system. Files: `overview` (mission, roadmap), `baselines` (scope, stack, floors to beat), `experiments` (append-only run log), `decisions`, `setup`, `open-questions`.

Rules:

- Every file has frontmatter: `title`, `updated`, `status` (`current`/`stale`).
- One fact, one home — link with `[[wikilinks]]` instead of duplicating. Deep detail lives in `docs/` and `experiments/README.md`; the vault distills and points.
- Delete stale content instead of appending updates; git holds history. Bump `updated` when you edit.
- Exception: `notes/experiments.md` rows are **never** deleted — failed runs with takeaways are the point.
- Do **not** create new files in `notes/` without asking. Split a file only past ~500 lines.
- Write valid Obsidian-flavored markdown — use the `obsidian-markdown` skill in `.claude/skills/`.

Agent behavior:

- Update the relevant note **in the same commit** as the code change it describes (or delegate to the `vault-keeper` agent with a session summary).
- When a milestone lands, flip its status in the roadmap table in `notes/overview.md`.
- Log every training/eval run as a row in `notes/experiments.md` automatically, with the short commit hash of the code that ran.
- Record every tradeoff/choice in `notes/decisions.md`; resolve `notes/open-questions.md` items as answers land.
- `notes/setup.md` must reflect the machine's *current* state — update it whenever something gets installed.

## Hard project rules (details in the vault)

- **Robot truth** is `configs/robots/a3_ultra.yaml` — never index qpos directly; go through `MujocoRobot`. Official vs ASSUMED values are marked; don't blur them.
- **`assets/a3_ultra/holosoma/*` is generated** by `scripts/convert/make_holosoma_asset.py` — never hand-edit; regenerate.
- **Default simulator is `isaacsim`** — presets, `scripts/train/train_a3_wsl.sh`, and the cloud script all default to it. Reach for `simulator:mjwarp` only for explicit local smoke runs.
- **mujoco_warp is pinned** to git `ecaef88` (PyPI 3.11.0 breaks physics). Never "upgrade" it casually — applies to the smoke path only.
- Judge policies against the recorded floors (`notes/baselines.md`), not against feel.
