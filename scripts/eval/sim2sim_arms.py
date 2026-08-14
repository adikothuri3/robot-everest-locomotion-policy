"""Does the locomotion policy survive an upper-body skill moving the arms?

The trained policy owns all 29 DOF, and its `pose` reward weights the 17
waist+arm joints at 50.0 against 0.01-5.0 for the legs — which is why the arms
hang at the sides in every showcase clip. So a manipulation skill does not "add"
arm motion; it *overrides* 14 of the policy's 29 outputs and drives the arms to
joint positions the policy never saw during training.

This script measures what that costs. A scripted arm trajectory replaces the
policy's arm targets; the policy still observes the true arm state (it sees what
the skill did) but cannot undo it. Everything else — legs, waist, terrain,
commands — is unchanged, and each trajectory is compared against the identical
run with the arms left at default.

    python scripts/eval/sim2sim_arms.py --onnx <policy>.onnx
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402

from everest_locomotion.robots.manifest import load_manifest  # noqa: E402
from everest_locomotion.terrains import procedural_rough  # noqa: E402
from everest_locomotion.evaluation.sim2sim import (  # noqa: E402
    A3Sim,
    Command,
    HolosomaPolicy,
    build_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim2sim_suite import rough  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

ARM_JOINTS = [
    f"{side}_{j}"
    for side in ("left", "right")
    for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
              "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
]


@dataclass
class ArmSkill:
    """A scripted upper-body motion, in absolute joint targets."""

    name: str
    description: str
    #: joint name -> f(t) target in rad. Unlisted arm joints hold their default.
    joints: dict = field(default_factory=dict)
    ramp_s: float = 1.0        # ease in from the default pose, so t=0 is not a step
    start_s: float = 2.0       # let the gait settle before the skill engages

    def make_override(self, sim: A3Sim):
        names = sim.policy.dof_names
        idx = {n: i for i, n in enumerate(names)}
        default = sim.policy.default_dof_pos
        targets = {idx[n]: f for n, f in self.joints.items()}
        arm_idx = [idx[n] for n in ARM_JOINTS]

        def override(target: np.ndarray, t: float) -> np.ndarray:
            out = target.copy()
            # the skill owns every arm joint, holding default unless scripted
            out[arm_idx] = default[arm_idx]
            if t < self.start_s:
                return out
            blend = min(1.0, (t - self.start_s) / self.ramp_s) if self.ramp_s > 0 else 1.0
            for i, f in targets.items():
                out[i] = default[i] + blend * (f(t - self.start_s) - default[i])
            return out

        return override


def both(joint: str, fn):
    """Same trajectory on the left and right arm."""
    return {f"left_{joint}": fn, f"right_{joint}": fn}


def const(v):
    return lambda t: v


SKILLS = [
    ArmSkill("arms_at_default", "control — policy keeps the arms (as shipped)", {}),
    ArmSkill(
        "reach_forward", "both arms reached out in front, held",
        {**both("shoulder_pitch_joint", const(-1.4)), **both("elbow_joint", const(0.9))},
    ),
    ArmSkill(
        "raise_overhead", "both arms raised overhead — biggest CoM shift",
        {**both("shoulder_pitch_joint", const(-2.6))},
    ),
    ArmSkill(
        "reach_sideways", "both arms out to the sides (T-pose)",
        {"left_shoulder_roll_joint": const(1.5), "right_shoulder_roll_joint": const(-1.5)},
    ),
    ArmSkill(
        "asymmetric_reach", "one arm forward and across — asymmetric load",
        {"left_shoulder_pitch_joint": const(-1.5), "left_shoulder_roll_joint": const(1.2),
         "left_elbow_joint": const(1.2)},
    ),
    ArmSkill(
        "swing_slow", "arms swinging 0.5 Hz, ±0.8 rad (natural counter-swing)",
        {"left_shoulder_pitch_joint": lambda t: -0.8 * math.sin(2 * math.pi * 0.5 * t),
         "right_shoulder_pitch_joint": lambda t: 0.8 * math.sin(2 * math.pi * 0.5 * t)},
    ),
    ArmSkill(
        "swing_fast", "arms swinging 1.5 Hz, ±1.0 rad (aggressive, off-distribution)",
        {"left_shoulder_pitch_joint": lambda t: -1.0 * math.sin(2 * math.pi * 1.5 * t),
         "right_shoulder_pitch_joint": lambda t: 1.0 * math.sin(2 * math.pi * 1.5 * t)},
    ),
    ArmSkill(
        "stir_both", "both arms circling 1.0 Hz — continuous angular momentum injection",
        {"left_shoulder_pitch_joint": lambda t: -1.0 + 0.9 * math.sin(2 * math.pi * t),
         "left_shoulder_roll_joint": lambda t: 0.6 + 0.6 * math.cos(2 * math.pi * t),
         "right_shoulder_pitch_joint": lambda t: -1.0 + 0.9 * math.sin(2 * math.pi * t + 1.0),
         "right_shoulder_roll_joint": lambda t: -0.6 - 0.6 * math.cos(2 * math.pi * t + 1.0)},
    ),
]

CONTEXTS = [
    ("stand", Command(), None),
    ("walk_0.5", Command(lin_vel_x=0.5), None),
    ("walk_1.0", Command(lin_vel_x=1.0), None),
    ("rough_walk", Command(lin_vel_x=0.5), "rough"),
]


def mask_arm_obs(sim: A3Sim):
    """Hide the moved arms from the policy: report arm dof_pos/dof_vel as default.

    Decisive ablation. The policy owns 29 DOF but its `pose` reward pinned the arms
    near default in training, so it has effectively never seen a large value on those
    observation channels. If hiding them rescues the failing skills, the failure is
    an off-distribution *observation*, not a balance problem — and the fix is cheap.
    Layout (see evaluation.sim2sim): dof_pos is obs[37:66], dof_vel is obs[66:95],
    both in canonical dof order, so the 14 arm joints are the last 14 of each block.
    """
    n = sim.policy.n_dof
    arm0 = n - 14                       # arms are the final 14 canonical DOF
    # actions(n) + ang_vel(3) + cmd_ang(1) + cmd_lin(2) + cos_phase(2) = n + 8
    pos_lo, vel_lo = n + 8, 2 * n + 8

    def transform(obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        out[pos_lo + arm0: pos_lo + n] = 0.0   # dof_pos is already default-relative
        out[vel_lo + arm0: vel_lo + n] = 0.0
        return out

    return transform


def arm_mass_report(sim: A3Sim) -> dict:
    m = sim.model
    total = float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")])
    arms = 0.0
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_Link")
        arms += float(m.body_subtreemass[bid])
    return {"total_kg": round(total, 2), "arms_kg": round(arms, 2),
            "arms_frac": round(arms / total, 4)}


def amplitude_sweep(policy, manifest, duration_s: float) -> dict:
    """How far can a skill move the arms before the unmasked policy loses the legs?

    Ramps a held two-arm reach from 0 rad of shoulder pitch upward and reports the
    largest amplitude that still survives — the operating limit for a caller that
    cannot mask the arm observation channels.
    """
    # Shoulder pitch alone, then pitch combined with elbow flexion: `reach_forward`
    # (pitch 1.4 + elbow 0.9) falls while pitch 1.4 alone does not, so the axis that
    # matters is how many arm channels are off-distribution at once, not one angle.
    families = {
        "pitch_only": lambda a: both("shoulder_pitch_joint", const(-a)),
        "pitch_plus_elbow": lambda a: {**both("shoulder_pitch_joint", const(-a)),
                                       **both("elbow_joint", const(0.65 * a))},
    }
    amps = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8)
    out = {}
    for ctx_name, cmd in (("stand", Command()), ("walk_0.5", Command(lin_vel_x=0.5))):
        for fam, build in families.items():
            last_ok = 0.0
            for amp in amps:
                skill = ArmSkill(f"{fam}_{amp:g}", "", build(amp))
                sim = A3Sim(policy, manifest)
                res = sim.run(duration_s=duration_s, command=cmd,
                              target_override=skill.make_override(sim))
                print(f"  {ctx_name:9s} {fam:17s} {amp:.1f} rad  "
                      f"{'OK  ' if res.survived else 'FALL'}  tilt={res.max_tilt_deg:5.1f}")
                if not res.survived:
                    break
                last_ok = amp
            out[f"{ctx_name}/{fam}"] = last_ok
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--name", default="arm_skills")
    p.add_argument("--mask-arm-obs", action="store_true",
                   help="hide the moved arms from the policy's observation (ablation)")
    p.add_argument("--amplitude-sweep", action="store_true",
                   help="find the largest survivable arm deviation, unmasked")
    args = p.parse_args()

    manifest = load_manifest()
    policy = HolosomaPolicy(args.onnx)

    if args.amplitude_sweep:
        print("amplitude sweep (unmasked, held two-arm reach):")
        limits = amplitude_sweep(policy, manifest, args.duration)
        path = REPO / "results" / "sim2sim" / "arm_amplitude_limit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(limits, indent=2))
        print(f"\nmax survivable shoulder pitch: {limits}\nresults: {path}")
        return

    rows = []
    mass = None
    for ctx_name, cmd, terrain in CONTEXTS:
        patch = procedural_rough(rough(0.5)) if terrain else None
        model = build_model(patch)
        for skill in SKILLS:
            sim = A3Sim(policy, manifest, model=model, patch=patch)
            if mass is None:
                mass = arm_mass_report(sim)
            res = sim.run(
                duration_s=args.duration,
                command=cmd,
                name=f"{ctx_name}/{skill.name}",
                target_override=skill.make_override(sim),
                obs_transform=mask_arm_obs(sim) if args.mask_arm_obs else None,
            )
            rows.append({"context": ctx_name, "skill": skill.name,
                         "description": skill.description, **res.as_row()})
            status = "OK  " if res.survived else "FALL"
            print(f"[{status}] {ctx_name:11s} {skill.name:17s} "
                  f"velerr={res.lin_vel_error:6.3f} tilt={res.max_tilt_deg:5.1f} "
                  f"jit={res.action_jitter:.3f} "
                  f"fall={'-' if res.fall_time_s is None else f'{res.fall_time_s:.1f}s'}")

    # degradation vs the arms-at-default control, per context
    summary = {"arm_mass": mass, "by_context": {}}
    for ctx_name, _, _ in CONTEXTS:
        ctx = [r for r in rows if r["context"] == ctx_name]
        base = next(r for r in ctx if r["skill"] == "arms_at_default")
        moved = [r for r in ctx if r["skill"] != "arms_at_default"]
        summary["by_context"][ctx_name] = {
            "n_skills": len(moved),
            "n_survived": sum(r["survived"] for r in moved),
            "baseline_lin_vel_error": base["lin_vel_error"],
            "baseline_max_tilt_deg": base["max_tilt_deg"],
            "worst_lin_vel_error": max(r["lin_vel_error"] for r in moved),
            "worst_max_tilt_deg": max(r["max_tilt_deg"] for r in moved),
            "failures": [r["skill"] for r in moved if not r["survived"]],
        }

    n_ok = sum(r["survived"] for r in rows if r["skill"] != "arms_at_default")
    n_tot = sum(1 for r in rows if r["skill"] != "arms_at_default")
    out = REPO / "results" / "sim2sim" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\narms moved by a skill: survived {n_ok}/{n_tot}")
    print(f"arm mass {mass['arms_kg']} kg of {mass['total_kg']} kg "
          f"({100 * mass['arms_frac']:.1f}% of body mass)")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
