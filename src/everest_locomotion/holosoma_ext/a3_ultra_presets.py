"""Holosoma `--import-file` presets for the AgiBot A3 Ultra (29-DOF locomotion).

Usage (inside the Holosoma env, e.g. WSL2):
    python src/holosoma/holosoma/train_agent.py exp:a3-ultra-fast-sac simulator:mjwarp \
        --import-file /mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy/src/everest_locomotion/holosoma_ext/a3_ultra_presets.py

Design notes:
- Joint naming: identical convention to Unitree G1 29-DOF (verified), so all
  G1 locomotion presets (observation/action/termination/randomization/command/
  curriculum/reward) are robot-agnostic and reused unmodified. Only the
  RobotConfig is A3-specific. This keeps the first port faithful to upstream
  (Phase 8 rule: do NOT redesign rewards yet).
- Limits/efforts/velocities: official values from AgibotTech/A3-A3U-robot-model
  @ 589f508 (see configs/robots/a3_ultra.yaml for provenance).
- PD gains, armature, init pose: ASSUMED (documented in the manifest); tune.
- Asset: generated 29-DOF variant (head welded, foot contact points added) by
  scripts/convert/make_holosoma_asset.py.
"""

import os
from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotConfig,
    RobotControlConfig,
    RobotInitState,
)
from holosoma.config_types.scene import PhysicsConfig, PhysXPhysicsConfig
from holosoma.config_values import simulator
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.robot import ROBOT_REGISTRY, g1_29dof as g1_robot

ASSET_ROOT = os.environ.get(
    "EVEREST_A3_ASSET_ROOT",
    "/mnt/c/Users/Aditya/VSCode/robot-everest-locomotion-policy/assets/a3_ultra/holosoma",
)

_LEG_JOINTS = [
    "{}_hip_pitch_joint", "{}_hip_roll_joint", "{}_hip_yaw_joint",
    "{}_knee_joint", "{}_ankle_pitch_joint", "{}_ankle_roll_joint",
]
_ARM_JOINTS = [
    "{}_shoulder_pitch_joint", "{}_shoulder_roll_joint", "{}_shoulder_yaw_joint",
    "{}_elbow_joint", "{}_wrist_roll_joint", "{}_wrist_pitch_joint", "{}_wrist_yaw_joint",
]
DOF_NAMES = (
    [j.format("left") for j in _LEG_JOINTS]
    + [j.format("right") for j in _LEG_JOINTS]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [j.format("left") for j in _ARM_JOINTS]
    + [j.format("right") for j in _ARM_JOINTS]
)

# official limits (order = DOF_NAMES)
_LOWER = [
    -2.5133, -0.5236, -2.7227, -0.1222, -0.9076, -0.3491,   # left leg
    -2.5133, -1.4835, -2.7227, -0.1222, -0.9076, -0.3491,   # right leg
    -2.6180, -0.3491, -0.4887,                              # waist y/r/p
    -2.8798, -0.0873, -2.7925, -0.9599, -2.7925, -1.6232, -1.6232,  # left arm
    -2.8798, -2.6180, -2.7925, -0.9599, -2.7925, -1.6232, -1.6232,  # right arm
]
_UPPER = [
    2.9322, 1.4835, 2.7227, 2.5307, 0.5236, 0.3491,
    2.9322, 0.5236, 2.7227, 2.5307, 0.5236, 0.3491,
    2.6180, 0.3491, 0.4189,
    2.8798, 2.6180, 2.7925, 1.7453, 2.7925, 1.6232, 1.6232,
    2.8798, 0.0873, 2.7925, 1.7453, 2.7925, 1.6232, 1.6232,
]
_VEL = [
    12.043, 12.043, 12.043, 14.661, 10.8, 19.37,
    12.043, 12.043, 12.043, 14.661, 10.8, 19.37,
    12.043, 22.7, 9.248,
    13.614, 13.614, 15.708, 15.708, 15.708, 12.776, 12.776,
    13.614, 13.614, 15.708, 15.708, 15.708, 12.776, 12.776,
]
_EFFORT = [
    220.0, 220.0, 220.0, 320.0, 118.2, 54.75,
    220.0, 220.0, 220.0, 320.0, 118.2, 54.75,
    220.0, 46.0, 118.0,
    60.0, 60.0, 24.0, 24.0, 24.0, 6.0, 6.0,
    60.0, 60.0, 24.0, 24.0, 24.0, 6.0, 6.0,
]
# ASSUMED armature (no official spec); scaled by actuator class
_ARMATURE = [
    0.025, 0.025, 0.025, 0.025, 0.01, 0.01,
    0.025, 0.025, 0.025, 0.025, 0.01, 0.01,
    0.025, 0.01, 0.02,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.003,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.003,
]

BODY_NAMES = [
    "pelvis_link",
    "waist_yaw_Link", "waist_roll_Link", "torso_Link",
    "head_yaw_Link", "head_pitch_Link",
    "left_shoulder_pitch_Link", "left_shoulder_roll_Link", "left_shoulder_yaw_Link",
    "left_elbow_Link", "left_wrist_roll_Link", "left_wrist_pitch_Link", "left_wrist_yaw_Link",
    "right_shoulder_pitch_Link", "right_shoulder_roll_Link", "right_shoulder_yaw_Link",
    "right_elbow_Link", "right_wrist_roll_Link", "right_wrist_pitch_Link", "right_wrist_yaw_Link",
    "left_hip_pitch_Link", "left_hip_roll_Link", "left_hip_yaw_Link", "left_knee_Link",
    "left_ankle_pitch_Link", "left_ankle_roll_Link", "left_foot_contact_point",
    "right_hip_pitch_Link", "right_hip_roll_Link", "right_hip_yaw_Link", "right_knee_Link",
    "right_ankle_pitch_Link", "right_ankle_roll_Link", "right_foot_contact_point",
]

DEFAULT_JOINT_ANGLES = {name: 0.0 for name in DOF_NAMES}
DEFAULT_JOINT_ANGLES.update(
    {
        "left_hip_pitch_joint": -0.2, "left_knee_joint": 0.4, "left_ankle_pitch_joint": -0.2,
        "right_hip_pitch_joint": -0.2, "right_knee_joint": 0.4, "right_ankle_pitch_joint": -0.2,
        "left_shoulder_roll_joint": 0.15, "right_shoulder_roll_joint": -0.15,
        "left_elbow_joint": 0.3, "right_elbow_joint": 0.3,
    }
)

a3_ultra_29dof = RobotConfig(
    num_bodies=34,
    dof_obs_size=29,
    actions_dim=29,
    policy_obs_dim=-1,
    critic_obs_dim=-1,
    algo_obs_dim_dict={},
    key_bodies=["left_foot_contact_point", "right_foot_contact_point"],
    num_feet=2,
    foot_body_name="ankle_roll_Link",
    foot_height_name="foot_contact_point",
    knee_name="knee_Link",
    torso_name="torso_Link",
    dof_names=DOF_NAMES,
    upper_dof_names=g1_robot.upper_dof_names,
    upper_left_arm_dof_names=g1_robot.upper_left_arm_dof_names,
    upper_right_arm_dof_names=g1_robot.upper_right_arm_dof_names,
    lower_dof_names=g1_robot.lower_dof_names,
    has_torso=True,
    has_upper_body_dof=True,
    left_ankle_dof_names=g1_robot.left_ankle_dof_names,
    right_ankle_dof_names=g1_robot.right_ankle_dof_names,
    knee_dof_names=g1_robot.knee_dof_names,
    hips_dof_names=g1_robot.hips_dof_names,
    dof_pos_lower_limit_list=_LOWER,
    dof_pos_upper_limit_list=_UPPER,
    dof_vel_limit_list=_VEL,
    dof_effort_limit_list=_EFFORT,
    dof_armature_list=_ARMATURE,
    dof_joint_friction_list=[0.0] * 29,
    body_names=BODY_NAMES,
    terminate_after_contacts_on=["pelvis", "shoulder", "hip"],
    penalize_contacts_on=["pelvis", "shoulder", "hip"],
    init_state=RobotInitState(
        pos=[0.0, 0.0, 1.07],
        rot=[0.0, 0.0, 0.0, 1.0],
        lin_vel=[0.0, 0.0, 0.0],
        ang_vel=[0.0, 0.0, 0.0],
        default_joint_angles=DEFAULT_JOINT_ANGLES,
    ),
    randomize_link_body_names=[
        "pelvis_link",
        "left_hip_pitch_Link", "left_hip_roll_Link", "left_hip_yaw_Link", "left_knee_Link",
        "right_hip_pitch_Link", "right_hip_roll_Link", "right_hip_yaw_Link", "right_knee_Link",
    ],
    waist_dof_names=g1_robot.waist_dof_names,
    waist_yaw_dof_name="waist_yaw_joint",
    waist_roll_dof_name="waist_roll_joint",
    waist_pitch_dof_name="waist_pitch_joint",
    arm_dof_names=g1_robot.arm_dof_names,
    left_arm_dof_names=g1_robot.left_arm_dof_names,
    right_arm_dof_names=g1_robot.right_arm_dof_names,
    symmetry_joint_names=g1_robot.symmetry_joint_names,
    flip_sign_joint_names=g1_robot.flip_sign_joint_names,
    apply_dof_armature_in_isaacgym=True,
    contact_pairs_multiplier=16,
    control=RobotControlConfig(
        control_type="P",
        # ASSUMED gains for a 60 kg humanoid (see configs/robots/a3_ultra.yaml)
        stiffness={
            "hip_yaw": 200.0, "hip_roll": 200.0, "hip_pitch": 200.0,
            "knee": 200.0,
            "ankle_pitch": 100.0, "ankle_roll": 100.0,
            "waist_yaw": 300.0, "waist_roll": 300.0, "waist_pitch": 300.0,
            "shoulder_pitch": 60.0, "shoulder_roll": 60.0, "shoulder_yaw": 60.0,
            "elbow": 60.0,
            "wrist_roll": 10.0, "wrist_pitch": 10.0, "wrist_yaw": 10.0,
        },
        damping={
            "hip_yaw": 5.0, "hip_roll": 5.0, "hip_pitch": 5.0,
            "knee": 5.0,
            "ankle_pitch": 4.0, "ankle_roll": 4.0,
            "waist_yaw": 6.0, "waist_roll": 6.0, "waist_pitch": 6.0,
            "shoulder_pitch": 2.0, "shoulder_roll": 2.0, "shoulder_yaw": 2.0,
            "elbow": 2.0,
            "wrist_roll": 0.5, "wrist_pitch": 0.5, "wrist_yaw": 0.5,
        },
        action_scale=0.25,
        action_clip_value=100.0,
        clip_actions=True,
        clip_torques=True,
    ),
    asset=RobotAssetConfig(
        asset_root=ASSET_ROOT,
        collapse_fixed_joints=True,
        replace_cylinder_with_capsule=True,
        flip_visual_attachments=False,
        armature=0.001,
        thickness=0.01,
        urdf_file="a3_ultra_29dof.urdf",
        usd_file=None,
        xml_file="a3_ultra_29dof.xml",
        robot_type="a3_ultra_29dof",
        enable_self_collisions=False,
        default_dof_drive_mode=3,
        fix_base_link=False,
        link_physics=PhysicsConfig(
            physx=PhysXPhysicsConfig(
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
        ),
    ),
)

ROBOT_REGISTRY.add("a3_ultra_29dof", a3_ultra_29dof)

# Experiments: G1 recipes with the A3 robot swapped in (faithful port).
a3_ultra_fast_sac = EXPERIMENT_REGISTRY.add(
    "a3_ultra_fast_sac",
    replace(
        g1_29dof_fast_sac,
        training=TrainingConfig(project="everest-a3", name="a3_ultra_fast_sac"),
        robot=a3_ultra_29dof,
        simulator=simulator.mjwarp,
    ),
)

a3_ultra_ppo = EXPERIMENT_REGISTRY.add(
    "a3_ultra_ppo",
    replace(
        g1_29dof,
        training=TrainingConfig(project="everest-a3", name="a3_ultra_ppo"),
        robot=a3_ultra_29dof,
        simulator=simulator.mjwarp,
    ),
)

# ---------------------------------------------------------------------------
# Everest stability variant (E06): upstream FastSAC reward + explicit
# stability terms. Objective hierarchy: don't fall > recover > don't slip >
# reliable footholds > posture > tracking > speed. Action-rate penalty is NOT
# increased (aggressive recovery must stay possible).
# ---------------------------------------------------------------------------
from holosoma.config_types.reward import RewardTermCfg  # noqa: E402
from holosoma.config_values.loco.g1.reward import g1_29dof_loco_fast_sac  # noqa: E402

_everest_terms = dict(g1_29dof_loco_fast_sac.terms)
_everest_terms.update(
    {
        "penalty_stumble": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stumble",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_foothold": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_foothold",
            weight=-1.0,
            params={"foothold_epsilon": 0.01},
            tags=["penalty_curriculum"],
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-10.0,
            params={"soft_dof_pos_limit": 0.95},
        ),
        # do-not-fall dominates tracking
        "alive": replace(_everest_terms["alive"], weight=15.0)
        if "alive" in _everest_terms
        else RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive", weight=15.0, params={}
        ),
    }
)
a3_ultra_reward_everest = replace(g1_29dof_loco_fast_sac, terms=_everest_terms)

a3_ultra_fast_sac_everest = EXPERIMENT_REGISTRY.add(
    "a3_ultra_fast_sac_everest",
    replace(
        a3_ultra_fast_sac,
        training=TrainingConfig(project="everest-a3", name="a3_ultra_fast_sac_everest"),
        reward=a3_ultra_reward_everest,
    ),
)
