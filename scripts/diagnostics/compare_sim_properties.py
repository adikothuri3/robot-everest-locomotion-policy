"""Compare two dump_model_properties.py JSON files and report inconsistencies.

    python scripts/diagnostics/compare_sim_properties.py results/consistency/mujoco.json results/consistency/isaac.json
"""

from __future__ import annotations

import json
import sys

TOL_MASS = 0.01       # kg per body
TOL_RANGE = 0.002     # rad
TOL_TOTAL = 0.05      # kg


def norm(name: str) -> str:
    return name.lower()


def main() -> int:
    a = json.loads(open(sys.argv[1]).read())
    b = json.loads(open(sys.argv[2]).read())
    issues = []

    if abs(a["total_mass"] - b["total_mass"]) > TOL_TOTAL:
        issues.append(f"total mass: {a['total_mass']:.3f} vs {b['total_mass']:.3f}")
    if a["n_actuated"] != b["n_actuated"]:
        issues.append(f"actuated joints: {a['n_actuated']} vs {b['n_actuated']}")

    ja = {norm(k): v for k, v in a["joints"].items()}
    jb = {norm(k): v for k, v in b["joints"].items()}
    for name in sorted(set(ja) | set(jb)):
        if name not in ja or name not in jb:
            issues.append(f"joint only in one sim: {name}")
            continue
        ra, rb = ja[name]["range"], jb[name]["range"]
        if abs(ra[0] - rb[0]) > TOL_RANGE or abs(ra[1] - rb[1]) > TOL_RANGE:
            issues.append(f"joint range {name}: {ra} vs {rb}")

    ba = {norm(k): v for k, v in a["bodies"].items()}
    bb = {norm(k): v for k, v in b["bodies"].items()}
    for name in sorted(set(ba) & set(bb)):
        if abs(ba[name]["mass"] - bb[name]["mass"]) > TOL_MASS:
            issues.append(f"body mass {name}: {ba[name]['mass']:.4f} vs {bb[name]['mass']:.4f}")

    print(f"compared {a['simulator']} vs {b['simulator']}")
    if issues:
        print(f"{len(issues)} INCONSISTENCIES:")
        for i in issues:
            print("  -", i)
        return 1
    print("CONSISTENT within tolerances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
