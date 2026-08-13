"""Canonical robot manifest loader.

The YAML manifest (configs/robots/<name>.yaml) is the single source of truth for
joint ordering, limits, default pose, and PD gains. Simulator adapters must map
their internal ordering to/from the manifest's `joint_order`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from everest_locomotion import REPO_ROOT


@dataclass
class RobotManifest:
    raw: dict = field(repr=False)
    name: str
    joint_order: list[str]
    joints: dict[str, dict]
    control_joints: list[str]
    frozen_joints: list[str]
    default_joint_pos: dict[str, float]
    base_height: float
    mjcf_path: Path
    urdf_path: Path
    policy_frequency_hz: float
    sim_timestep_s: float
    decimation: int
    action_scale: float
    feet_bodies: list[str]
    base_body: str
    termination_contact_bodies: list[str]
    total_mass_kg: float

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)

    @property
    def num_actions(self) -> int:
        return len(self.control_joints)

    def joint_limits(self, joints: list[str] | None = None) -> np.ndarray:
        """(n, 2) array of [lower, upper] in manifest order."""
        joints = joints or self.joint_order
        return np.array([self.joints[j]["limits_rad"] for j in joints])

    def effort_limits(self, joints: list[str] | None = None) -> np.ndarray:
        joints = joints or self.joint_order
        return np.array([self.joints[j]["effort_nm"] for j in joints])

    def velocity_limits(self, joints: list[str] | None = None) -> np.ndarray:
        joints = joints or self.joint_order
        return np.array([self.joints[j]["vel_rad_s"] for j in joints])

    def default_pose_vector(self, joints: list[str] | None = None) -> np.ndarray:
        joints = joints or self.joint_order
        return np.array([self.default_joint_pos.get(j, 0.0) for j in joints])

    def pd_gains(self, joints: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the assumed PD gain groups to per-joint (kp, kd) vectors."""
        joints = joints or self.joint_order
        cfg = self.raw["pd_gains_assumed"]
        pattern_map: dict[str, str] = cfg["joint_pattern_map"]
        kp, kd = [], []
        for j in joints:
            group = None
            for pattern, g in pattern_map.items():
                if re.fullmatch(pattern, j):
                    group = g
                    break
            if group is None:
                raise ValueError(f"no PD gain group matches joint {j!r}")
            kp.append(cfg[group]["kp"])
            kd.append(cfg[group]["kd"])
        return np.array(kp), np.array(kd)


def load_manifest(name: str = "a3_ultra") -> RobotManifest:
    path = REPO_ROOT / "configs" / "robots" / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    robot = raw["robot"]
    control = raw["control"]
    joint_order = raw["joint_order"]
    frozen = control.get("frozen_joints", [])
    control_joints = [j for j in joint_order if j not in frozen]

    return RobotManifest(
        raw=raw,
        name=robot["name"],
        joint_order=joint_order,
        joints=raw["joints"],
        control_joints=control_joints,
        frozen_joints=frozen,
        default_joint_pos=raw["default_pose"]["joint_pos"],
        base_height=raw["default_pose"]["base_height_m"],
        mjcf_path=REPO_ROOT / robot["mjcf"],
        urdf_path=REPO_ROOT / robot["urdf"],
        policy_frequency_hz=control["policy_frequency_hz"],
        sim_timestep_s=control["sim_timestep_s"],
        decimation=control["decimation"],
        action_scale=control["action_scale"],
        feet_bodies=raw["bodies"]["feet"],
        base_body=raw["bodies"]["base"],
        termination_contact_bodies=raw["bodies"]["termination_contact_bodies"],
        total_mass_kg=robot["total_mass_kg"],
    )
