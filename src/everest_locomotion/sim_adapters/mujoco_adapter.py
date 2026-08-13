"""MuJoCo adapter for the A3 Ultra.

Maps the canonical manifest joint order onto MuJoCo qpos/qvel/actuator indices
and provides software PD torque control (the official MJCF ships raw torque
`<motor>` actuators with no gains).
"""

from __future__ import annotations

import numpy as np
import mujoco

from everest_locomotion.robots.manifest import RobotManifest


class MujocoRobot:
    def __init__(self, manifest: RobotManifest, model: mujoco.MjModel | None = None):
        self.manifest = manifest
        self.model = model or mujoco.MjModel.from_xml_path(str(manifest.mjcf_path))
        self.data = mujoco.MjData(self.model)

        m = self.model
        # canonical joint -> mujoco ids
        self.joint_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in manifest.joint_order]
        )
        if (self.joint_ids < 0).any():
            missing = [j for j, i in zip(manifest.joint_order, self.joint_ids) if i < 0]
            raise ValueError(f"joints missing from MJCF: {missing}")
        self.qpos_adr = np.array([m.jnt_qposadr[i] for i in self.joint_ids])
        self.qvel_adr = np.array([m.jnt_dofadr[i] for i in self.joint_ids])

        # canonical joint -> actuator id (actuators are named after joints)
        self.act_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in manifest.joint_order]
        )
        if (self.act_ids < 0).any():
            missing = [j for j, i in zip(manifest.joint_order, self.act_ids) if i < 0]
            raise ValueError(f"actuators missing from MJCF: {missing}")

        self.kp, self.kd = manifest.pd_gains()
        self.effort_limit = manifest.effort_limits()
        self.default_pose = manifest.default_pose_vector()

        self.base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, manifest.base_body)
        self.feet_body_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in manifest.feet_bodies
        ]
        self.feet_geom_ids = [
            [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
            for b in self.feet_body_ids
        ]

    # ------------------------------------------------------------------
    def reset_to_default(self, base_height: float | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        h = base_height if base_height is not None else self.manifest.base_height
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = h
        self.data.qpos[3] = 1.0  # unit quaternion w
        self.data.qpos[self.qpos_adr] = self.default_pose
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    def pd_torque(self, target_pos: np.ndarray) -> np.ndarray:
        tau = self.kp * (target_pos - self.joint_pos()) - self.kd * self.joint_vel()
        return np.clip(tau, -self.effort_limit, self.effort_limit)

    def apply_pd(self, target_pos: np.ndarray) -> None:
        self.data.ctrl[self.act_ids] = self.pd_torque(target_pos)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def foot_contact_forces(self) -> list[float]:
        """Total normal contact force magnitude per foot [N]."""
        totals = [0.0] * len(self.feet_body_ids)
        force = np.zeros(6)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            for i, fb in enumerate(self.feet_body_ids):
                if b1 == fb or b2 == fb:
                    mujoco.mj_contactForce(self.model, self.data, c, force)
                    totals[i] += abs(force[0])
        return totals

    def base_pos(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def projected_gravity(self) -> np.ndarray:
        """Gravity direction in base frame (z-up world: [0,0,-1] rotated)."""
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.base_quat_wxyz())
        return rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
