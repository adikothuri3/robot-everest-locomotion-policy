#!/bin/bash
# Peek at latest scalars in the newest holosoma tensorboard run.
cd ~/holosoma
source .venv/hsmujoco/bin/activate
python - <<'EOF'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(Path.home().glob("holosoma/logs/*/*/"))
run = runs[-1]
print("run:", run.name)
acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]
for tag in tags:
    ev = acc.Scalars(tag)
    if ev:
        print(f"{tag}: n={len(ev)} last_step={ev[-1].step} last={ev[-1].value:.4f}")
EOF
