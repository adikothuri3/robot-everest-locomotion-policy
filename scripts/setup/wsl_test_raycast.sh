#!/bin/bash
# Unit-test holosoma's patched warp ray_cast: flat quad at z=0.5, rays straight down.
set -e
cd ~/holosoma
source .venv/hsmujoco/bin/activate
python - <<'EOF'
import torch
import warp as wp
wp.init()
from holosoma.utils import warp_utils

points = wp.array(
    [[-10.0, -10.0, 0.5], [10.0, -10.0, 0.5], [10.0, 10.0, 0.5], [-10.0, 10.0, 0.5]],
    dtype=wp.vec3, device="cuda:0",
)
indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device="cuda:0")
mesh = wp.Mesh(points=points, indices=indices)

starts = torch.tensor([[0.0, 0.0, 5.0], [1.0, 1.0, 3.0], [-2.0, 0.5, 10.0]], device="cuda")
dirs = torch.tensor([[0.0, 0.0, -1.0]] * 3, device="cuda")
hits = warp_utils.ray_cast(starts, dirs, mesh)
print("hits:", hits.cpu().numpy())
expected_z = 0.5
ok = torch.allclose(hits[:, 2], torch.full((3,), expected_z, device="cuda"), atol=1e-4)
print("RAYCAST", "PASS" if ok else "FAIL")
EOF
