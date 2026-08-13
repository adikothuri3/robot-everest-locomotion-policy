#!/bin/bash
# Wait until the A3 run's tensorboard file has scalars, then dump reward metrics.
cd ~/holosoma
source .venv/hsmujoco/bin/activate
RUN=$(ls -d ~/holosoma/logs/everest-a3/*/ | sort | tail -1)
echo "run: $RUN"
for i in $(seq 1 60); do
  SZ=$(stat -c%s "$RUN"/events.out.tfevents.* 2>/dev/null | head -1)
  if [ -n "$SZ" ] && [ "$SZ" -gt 2000 ]; then break; fi
  sleep 20
done
python - "$RUN" <<'EOF'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
tags = sorted(acc.Tags()["scalars"])
print("n_tags:", len(tags))
for tag in tags:
    if any(k in tag for k in ("rew", "Episode", "env_rewards", "Time", "len", "fps")):
        ev = acc.Scalars(tag)
        print(f"{tag}: step={ev[-1].step} value={ev[-1].value:.4f}")
EOF
