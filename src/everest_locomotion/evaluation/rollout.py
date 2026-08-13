"""MuJoCo evaluation rollouts with stability metrics.

Runs a Policy at the manifest's control frequency on a MujocoRobot (optionally
on generated terrain), applies scripted disturbances, and records the metrics
defined by the stability benchmark (survival, slip, tracking, control health).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from everest_locomotion.policies.base import Policy
from everest_locomotion.sim_adapters.mujoco_adapter import MujocoRobot


@dataclass
class Push:
    time_s: float
    velocity_xyz: tuple[float, float, float]  # instantaneous base velocity change [m/s]


@dataclass
class RolloutResult:
    survived: bool
    fall_time_s: float | None
    duration_s: float
    mean_base_height: float
    max_tilt_deg: float
    slip_distance_m: float
    slip_events: int
    mean_abs_lin_vel_error: float | None
    torque_saturation_frac: float
    max_joint_acc: float
    action_jitter: float
    joint_limit_events: int
    extras: dict = field(default_factory=dict)


def _foot_slip(robot: MujocoRobot, dt: float) -> tuple[float, bool]:
    """Tangential foot displacement while in contact this step."""
    slip = 0.0
    slipping = False
    for i, bid in enumerate(robot.feet_body_ids):
        in_contact = any(
            robot.model.geom_bodyid[robot.data.contact[c].geom1] == bid
            or robot.model.geom_bodyid[robot.data.contact[c].geom2] == bid
            for c in range(robot.data.ncon)
        )
        if in_contact:
            v = robot.data.cvel[bid][3:5]  # world-frame linear vel x,y of body
            speed = float(np.linalg.norm(v))
            slip += speed * dt
            if speed > 0.1:
                slipping = True
    return slip, slipping


def run_rollout(
    robot: MujocoRobot,
    policy: Policy,
    duration_s: float = 10.0,
    pushes: list[Push] | None = None,
    command_vel_xy: tuple[float, float] | None = None,
    settle_s: float = 0.5,
    terrain_height_fn=None,
) -> RolloutResult:
    manifest = robot.manifest
    m, d = robot.model, robot.data
    dt = m.opt.timestep
    decimation = manifest.decimation
    pushes = sorted(pushes or [], key=lambda p: p.time_s)

    ctrl_idx = [manifest.joint_order.index(j) for j in manifest.control_joints]
    default_ctrl = robot.default_pose[ctrl_idx]

    base_h = manifest.base_height
    if terrain_height_fn is not None:
        base_h += terrain_height_fn(0.0, 0.0)
    robot.reset_to_default(base_height=base_h + 0.02)
    policy.reset()

    target_full = robot.default_pose.copy()
    n_steps = int(duration_s / dt)
    push_i = 0

    heights, tilts, torques_sat, jerks, slips = [], [], [], [], 0.0
    slip_event_count = 0
    was_slipping = False
    vel_errors = []
    max_acc = 0.0
    prev_action = default_ctrl.copy()
    prev_vel = robot.joint_vel()
    limit_events = 0
    fall_time = None
    limits = manifest.joint_limits()

    for step in range(n_steps):
        t = step * dt
        if step % decimation == 0:
            obs = {
                "base_ang_vel": d.qvel[3:6].copy(),
                "projected_gravity": robot.projected_gravity(),
                "joint_pos": robot.joint_pos()[ctrl_idx] - default_ctrl,
                "joint_vel": robot.joint_vel()[ctrl_idx],
                "prev_action": prev_action - default_ctrl,
                "command": np.array(
                    [command_vel_xy[0], command_vel_xy[1], 0.0] if command_vel_xy else [0, 0, 0]
                ),
            }
            action = policy.act(obs)
            jerks.append(float(np.abs(action - prev_action).mean()))
            prev_action = action
            target_full[ctrl_idx] = action

        while push_i < len(pushes) and t >= pushes[push_i].time_s + settle_s:
            d.qvel[0:3] += np.array(pushes[push_i].velocity_xyz)
            push_i += 1

        robot.apply_pd(target_full)
        robot.step()

        if not np.isfinite(d.qpos).all():
            fall_time = t
            break

        grav = robot.projected_gravity()
        tilt = np.degrees(np.arccos(np.clip(-grav[2], -1, 1)))
        tilts.append(tilt)
        z = robot.base_pos()[2]
        if terrain_height_fn is not None:
            z -= terrain_height_fn(*robot.base_pos()[:2])
        heights.append(z)
        if tilt > 60.0 or z < 0.4:
            fall_time = t
            break

        tau = d.ctrl[robot.act_ids]
        torques_sat.append(float(np.mean(np.abs(tau) >= 0.97 * robot.effort_limit)))
        vel = robot.joint_vel()
        max_acc = max(max_acc, float(np.abs((vel - prev_vel) / dt).max()))
        prev_vel = vel
        jp = robot.joint_pos()
        limit_events += int(((jp < limits[:, 0] + 0.01) | (jp > limits[:, 1] - 0.01)).sum() > 0)

        s, slipping = _foot_slip(robot, dt)
        slips += s
        if slipping and not was_slipping:
            slip_event_count += 1
        was_slipping = slipping

        if command_vel_xy is not None and t > settle_s:
            vel_errors.append(float(np.linalg.norm(d.qvel[0:2] - np.array(command_vel_xy))))

    return RolloutResult(
        survived=fall_time is None,
        fall_time_s=fall_time,
        duration_s=duration_s,
        mean_base_height=float(np.mean(heights)) if heights else 0.0,
        max_tilt_deg=float(np.max(tilts)) if tilts else 180.0,
        slip_distance_m=float(slips),
        slip_events=slip_event_count,
        mean_abs_lin_vel_error=float(np.mean(vel_errors)) if vel_errors else None,
        torque_saturation_frac=float(np.mean(torques_sat)) if torques_sat else 1.0,
        max_joint_acc=max_acc,
        action_jitter=float(np.mean(jerks)) if jerks else 0.0,
        joint_limit_events=limit_events,
    )
