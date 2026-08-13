"""Patch holosoma warp_utils.py for Warp >= 1.16 (wp.types.array removed).

Replaces the three zero-copy `wp.types.array(ptr=...)` constructions with
`wp.from_torch(..., dtype=wp.vec3)`, which is the supported API for wrapping
torch tensors. Idempotent. Run inside WSL:
    python3 patch_holosoma_warp.py ~/holosoma/src/holosoma/holosoma/utils/warp_utils.py
"""

import re
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
src = path.read_text()

if "wp.from_torch" in src:
    print("already patched")
    sys.exit(0)

pattern = re.compile(
    r"wp\.types\.array\(\s*ptr=(\w+)\.data_ptr\(\),\s*dtype=wp\.vec3,\s*shape=\(num_rays,\),\s*"
    r"copy=False,\s*(?:#[^\n]*\n\s*)?device=wp_mesh\.device,\s*\)",
    re.S,
)
new, n = pattern.subn(r"wp.from_torch(\1.contiguous(), dtype=wp.vec3)", src)
if n != 3:
    print(f"ERROR: expected 3 replacements, made {n}; aborting without write")
    sys.exit(1)
path.write_text(new)
print(f"patched {path} ({n} replacements)")
