"""Is the get-up terminal state reachable — or does the task forbid its own success?

`A3UltraGetupManager` spawns a fraction of its episodes standing at the
locomotion default pose, i.e. *exactly at* `_task_success_condition`; those only
have to hold it for `TASK_HOLD_STEPS`. Yet run E13 and the 2026-08-16 v3 run both
reported `getup_success_rate` **identically 0.0000** across 4096 envs. That is a
claim about the gate, not about how hard rising is.

This script settles it without training. It spawns standing exactly as
`PoseBankCommand` does, drives a real exported get-up policy, and scores the
trainer's own six-condition gate — sweeping the two things that differ between
the trainer and a deployed rollout:

  * **beta** (`getup_action_authority`), which multiplies every action. At 2.0
    the effective action scale is 0.5 instead of the deployable 0.25.
  * **exploration noise**, the std PPO samples actions with (`Policy/mean_noise_std`).

The finding: at **beta 2.0 the gate is unsatisfiable even with zero noise** —
doubling the action scale alone drives max |dof_vel| over the leg/waist joints to
~1.36 rad/s at a quiet stand, past `SUCCESS_JOINT_VEL` of 1.0. Since beta only
annealed when the success-linked rose proxy cleared its threshold, and beta itself
pinned that at zero, the curriculum could never move. The fix (implemented in the
extension) is a scheduled beta plus a leaky, EMA-smoothed hold counter; the
`old`/`new` columns below show both gates so the difference stays visible.

    python scripts/diagnostics/check_getup_terminal.py
    python scripts/diagnostics/check_getup_terminal.py --onnx checkpoints/<run>/model_X.onnx

Exit code is non-zero if the terminal state is unreachable under the *fixed*
gate at deploy authority with converged noise — i.e. if this regresses again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from everest_locomotion import REPO_ROOT
from everest_locomotion.evaluation.sim2sim import (
    ACTION_CLIP_VALUE,
    A3Sim,
    HolosomaPolicy,
)
from everest_locomotion.robots.manifest import load_manifest

# Verbatim from src/everest_locomotion/holosoma_ext/a3_ultra_getup.py — this
# script is only meaningful if it tests the *same* numbers the trainer scores.
STAND_BASE_HEIGHT = 1.063
UPRIGHT_GATE = -0.85
SUCCESS_HEIGHT = 0.98
SUCCESS_POSE_TOL = 0.30
SUCCESS_LIN_VEL = 0.25
SUCCESS_ANG_VEL = 0.8
SUCCESS_JOINT_VEL = 1.0
HOLD_STEPS = 100
SUCCESS_HOLD_DECAY = 2
JOINT_SPEED_TAU_S = 0.05
WRIST_ACTION_FACTOR = 0.2

DEFAULT_ONNX = REPO_ROOT / "checkpoints" / "v1_getup_28k_plateau" / "model_27999.onnx"


def probe(policy, manifest, beta: float, noise: float, seconds: float, seed: int) -> dict:
    """One standing-start episode; score the old and the fixed hold rule."""
    sim = A3Sim(policy, manifest)
    sim.action_authority = beta
    sim.wrist_action_factor = WRIST_ACTION_FACTOR
    rng = np.random.default_rng(seed)
    legs_waist = np.array(
        [i for i, n in enumerate(policy.dof_names)
         if any(k in n for k in ("hip", "knee", "ankle", "waist"))]
    )

    # PoseBankCommand's easy-start spawn: default pose, identity yaw,
    # STAND_BASE_HEIGHT + 0.005 (bank clearance) + 0.01 (PhysX clearance).
    mujoco.mj_resetData(sim.model, sim.data)
    sim.data.qpos[:] = 0.0
    sim.data.qpos[2] = STAND_BASE_HEIGHT + 0.015
    sim.data.qpos[3] = 1.0
    sim.data.qpos[sim.qpos_adr] = policy.default_dof_pos
    sim.data.qvel[:] = 0.0
    sim._obs_history = {}
    mujoco.mj_forward(sim.model, sim.data)

    dt = sim.control_dt
    alpha = dt / (JOINT_SPEED_TAU_S + dt)
    ema = np.zeros(policy.n_dof)
    prev_action = np.zeros(policy.n_dof)
    cmd = np.zeros(3)

    strict_run = strict_best = leaky = 0
    leaky_success = False
    passes = 0
    jv_raw_all = []
    scored = 0

    for step in range(int(round(seconds / dt))):
        obs = sim.observe(prev_action, step, cmd)
        mu = policy.act(obs)
        action = sim.apply_action(mu + rng.normal(0.0, noise, mu.shape))
        prev_action = action

        target = policy.default_dof_pos + policy.action_scale * np.clip(
            action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE
        )
        for _ in range(sim.decimation):
            q = sim.data.qpos[sim.qpos_adr]
            qd = sim.data.qvel[sim.qvel_adr]
            sim.data.ctrl[sim.act_ids] = np.clip(
                policy.kp * (target - q) - policy.kd * qd,
                -sim.effort_limit, sim.effort_limit,
            )
            mujoco.mj_step(sim.model, sim.data)
        if not np.isfinite(sim.data.qpos).all():
            break

        q = sim.data.qpos[sim.qpos_adr]
        qd = sim.data.qvel[sim.qvel_adr]
        ema += alpha * (np.abs(qd) - ema)

        base = (
            float(sim.base_pos()[2]) > SUCCESS_HEIGHT
            and float(sim.projected_gravity()[2]) < UPRIGHT_GATE
            and float(np.linalg.norm(sim.data.qvel[0:3])) < SUCCESS_LIN_VEL
            and float(np.linalg.norm(sim.data.qvel[3:6])) < SUCCESS_ANG_VEL
            and float(np.abs(q[legs_waist] - policy.default_dof_pos[legs_waist]).mean())
            < SUCCESS_POSE_TOL
        )
        jv_raw = float(np.abs(qd[legs_waist]).max())
        jv_ema = float(ema[legs_waist].max())
        jv_raw_all.append(jv_raw)
        scored += 1

        old_ok = base and jv_raw < SUCCESS_JOINT_VEL
        new_ok = base and jv_ema < SUCCESS_JOINT_VEL
        passes += int(new_ok)

        strict_run = strict_run + 1 if old_ok else 0
        strict_best = max(strict_best, strict_run)
        leaky = leaky + 1 if new_ok else max(0, leaky - SUCCESS_HOLD_DECAY)
        if leaky >= HOLD_STEPS:
            leaky_success = True

    jv = np.array(jv_raw_all) if jv_raw_all else np.zeros(1)
    return {
        "beta": beta,
        "noise": noise,
        "old_success": strict_best >= HOLD_STEPS,
        "old_best_run": strict_best,
        "new_success": leaky_success,
        "gate_pass": passes / max(1, scored),
        "jv_mean": float(jv.mean()),
        "jv_p95": float(np.percentile(jv, 95)),
    }


def check_feedback_convention(policy, manifest) -> bool:
    """Does the `actions` observation report the raw or the applied action?

    `ActionManager.process_actions` stores the tensor `_pre_physics_step` handed
    it, so an env that rescales before calling `super()` changes what the policy
    observes — but that is worth confirming against a trained policy, because
    the two conventions disagree sharply and picking wrong silently makes a
    working policy look broken. The discriminator is the standing anchor: the
    run's own `task_target_pose` telemetry (~= the easy-start share per step)
    says standing envs hold the default pose with near-zero error.
    """
    out = {}
    for convention in ("applied", "raw"):
        sim = A3Sim(policy, manifest)
        sim.wrist_action_factor = WRIST_ACTION_FACTOR
        mujoco.mj_resetData(sim.model, sim.data)
        sim.data.qpos[:] = 0.0
        sim.data.qpos[2] = STAND_BASE_HEIGHT + 0.015
        sim.data.qpos[3] = 1.0
        sim.data.qpos[sim.qpos_adr] = policy.default_dof_pos
        sim.data.qvel[:] = 0.0
        sim._obs_history = {}
        mujoco.mj_forward(sim.model, sim.data)
        prev = np.zeros(policy.n_dof)
        for step in range(500):
            raw = policy.act(sim.observe(prev, step, np.zeros(3)))
            applied = sim.apply_action(raw)
            prev = applied if convention == "applied" else raw
            target = policy.default_dof_pos + policy.action_scale * np.clip(
                applied, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE
            )
            for _ in range(sim.decimation):
                q = sim.data.qpos[sim.qpos_adr]
                qd = sim.data.qvel[sim.qvel_adr]
                sim.data.ctrl[sim.act_ids] = np.clip(
                    policy.kp * (target - q) - policy.kd * qd,
                    -sim.effort_limit, sim.effort_limit,
                )
                mujoco.mj_step(sim.model, sim.data)
        out[convention] = float(sim.base_pos()[2])
    print(f"feedback convention check (standing anchor, final pelvis): "
          f"applied {out['applied']:.3f} m, raw {out['raw']:.3f} m")
    if out["applied"] > SUCCESS_HEIGHT and out["raw"] < SUCCESS_HEIGHT:
        print("  -> applied-action feedback reproduces the run's telemetry, as assumed\n")
        return True
    print("  -> UNEXPECTED: the discriminator no longer separates the two "
          "conventions;\n     re-derive before trusting any grade below\n")
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=str(DEFAULT_ONNX))
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    onnx = Path(args.onnx)
    if not onnx.exists():
        raise SystemExit(f"no policy at {onnx} (see README 'Shipped policies')")
    policy = HolosomaPolicy(onnx)
    manifest = load_manifest()

    print(f"policy {onnx.name}   standing spawn (PoseBankCommand easy-start)")
    print(f"gate   height>{SUCCESS_HEIGHT} upright<{UPRIGHT_GATE} |v|<{SUCCESS_LIN_VEL} "
          f"|w|<{SUCCESS_ANG_VEL} pose<{SUCCESS_POSE_TOL} jvel<{SUCCESS_JOINT_VEL}, "
          f"held {HOLD_STEPS} steps")
    print("old = strictly-consecutive hold on instantaneous |dof_vel|  (the wedged run)")
    print(f"new = leaky hold (-{SUCCESS_HOLD_DECAY}/fail) on a {JOINT_SPEED_TAU_S:g}s EMA"
          "  (the fix)\n")
    convention_ok = check_feedback_convention(policy, manifest)
    print(f"{'beta':>5} {'noise':>6} | {'old hold':>9} {'old':>5} | "
          f"{'gate pass':>10} {'new':>5} | {'jvel mean':>10} {'jvel p95':>9}")

    grid = [
        (2.0, 0.00), (2.0, 0.10),          # the authority the wedged run sat at
        (1.0, 0.80), (1.0, 0.30),          # deploy authority, early exploration
        (1.0, 0.10), (1.0, 0.00),          # deploy authority, converged
    ]
    results = []
    for beta, noise in grid:
        r = probe(policy, manifest, beta, noise, args.duration, args.seed)
        results.append(r)
        print(f"{beta:5.1f} {noise:6.2f} | {r['old_best_run']:6d}/100 "
              f"{'YES' if r['old_success'] else 'no':>5} | {100 * r['gate_pass']:9.1f}% "
              f"{'YES' if r['new_success'] else 'no':>5} | "
              f"{r['jv_mean']:10.3f} {r['jv_p95']:9.3f}")

    deploy = next(r for r in results if r["beta"] == 1.0 and r["noise"] == 0.10)
    high_beta = next(r for r in results if r["beta"] == 2.0 and r["noise"] == 0.10)
    thrashing = next(r for r in results if r["beta"] == 1.0 and r["noise"] == 0.80)
    print()
    if high_beta["gate_pass"] < 0.05:
        print("CONFIRMED: at beta 2.0 the terminal gate is unreachable — the action "
              "authority alone,\n  not the policy, pins `getup_success_rate` at 0. "
              "beta must be scheduled, never gated\n  on a metric it suppresses.")
    if not deploy["new_success"]:
        print("REGRESSION: the fixed gate is unreachable at deploy authority with "
              "converged noise.\n  Training cannot register success — do not launch.")
        return 1
    if thrashing["new_success"]:
        print("WARNING: the fixed gate passes even for a thrashing policy "
              "(noise 0.8) — too loose.")
        return 1
    if not convention_ok:
        return 1
    print("OK: the fixed gate is reachable at deploy authority once exploration "
          "converges,\n  and still correctly unreachable for a thrashing policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
