"""HoST feasibility-probe config: A3 Ultra supine get-up (ground env).

Drop-in for InternRobotics/HoST `legged_gym/legged_gym/envs/a3/` — see
README.md in this folder for setup. Modeled 1:1 on their
`envs/g1/g1_config_ground.py` (G1Cfg / G1CfgPPO), with every robot-specific
value rescaled for the A3 Ultra (60.18 kg, 1.74 m, 29 DOF) using the scaling
heuristics from the HoST README:

  pull force  ~60% weight     -> 590 N * 0.6 / 2 trunk links = 175 each
  curriculum threshold height ~70% robot height -> 1.2 m (head)
  stage base heights: G1 0.45/0.45/0.65 at 0.75 stand -> x(1.063/0.75) = 0.64/0.64/0.92
  target head height ~75%     -> 1.30 m
  base_height_target: stand pelvis 1.063 -> 1.00 (HoST used ~95% of stand)

PURPOSE: throwaway probe. Answers "can this recipe discover ANY get-up at
60 kg with 24 Nm elbows?" — not the durable implementation (that's the
Holosoma extension). Expect reward-weight iteration; log torque traces.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Manifest-assumed PD gains (configs/robots/a3_ultra.yaml pd_gains_assumed).
# HoST G1 used stiffer arms (100); A3 arm motors are weaker (60/24/6 Nm) —
# start from manifest values, expect to revisit after first runs.
_STIFFNESS = {"hip": 200, "knee": 200, "ankle": 100, "waist": 300,
              "shoulder": 60, "elbow": 60, "wrist": 10}
_DAMPING = {"hip": 5, "knee": 5, "ankle": 4, "waist": 6,
            "shoulder": 2, "elbow": 2, "wrist": 0.5}

# Nominal stand = locomotion default_pose (the handoff target).
_TARGET_JOINT_ANGLES = {
    "left_hip_pitch_joint": -0.2, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.2, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.4, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.15,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.3,
    "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -0.15,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.3,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}


class A3UltraCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0.0, -1, 0, 1.0]  # supine drop, same as HoST G1 (xyzw, unnormalized)
        target_joint_angles = dict(_TARGET_JOINT_ANGLES)
        default_joint_angles = dict(_TARGET_JOINT_ANGLES)  # PD ref at zero action

    class env(LeggedRobotCfg.env):
        # HoST G1: 76 = 7 (ang_vel 3 + gravity 3 + 1) + 3*23. For 29 DOF: 7 + 87.
        # VERIFY against compute_observations() after cloning — see README.
        num_one_step_observations = 94
        num_actions = 29
        num_dofs = 29
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10
        unactuated_timesteps = 30

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = dict(_STIFFNESS)
        damping = dict(_DAMPING)
        action_scale = 1
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        measure_heights = True
        num_rows = 1
        num_cols = 20
        terrain_proportions = [1, 0.0, 0, 0, 0]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a3_ultra/a3_ultra_29dof.urdf"
        name = "a3_ultra"
        # substrings matched against A3 link names (left_ankle_roll_Link etc.)
        left_foot_name = "left_ankle_pitch"
        right_foot_name = "right_ankle_pitch"
        left_knee_name = "left_knee"
        right_knee_name = "right_knee"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["elbow", "shoulder", "waist", "knee", "hip"]
        terminate_after_contacts_on = []
        left_shoulder_name = "left_shoulder"
        right_shoulder_name = "right_shoulder"
        left_leg_joints = ["left_hip_yaw_joint", "left_hip_roll_joint",
                           "left_hip_pitch_joint", "left_knee_joint",
                           "left_ankle_pitch_joint", "left_ankle_roll_joint"]
        right_leg_joints = ["right_hip_yaw_joint", "right_hip_roll_joint",
                            "right_hip_pitch_joint", "right_knee_joint",
                            "right_ankle_pitch_joint", "right_ankle_roll_joint"]
        left_hip_joints = ["left_hip_yaw_joint"]
        right_hip_joints = ["right_hip_yaw_joint"]
        left_hip_roll_joints = ["left_hip_roll_joint"]
        right_hip_roll_joints = ["right_hip_roll_joint"]
        left_hip_pitch_joints = ["left_hip_pitch_joint"]
        right_hip_pitch_joints = ["right_hip_pitch_joint"]
        left_shoulder_roll_joints = ["left_shoulder_roll_joint"]
        right_shoulder_roll_joints = ["right_shoulder_roll_joint"]
        left_knee_joints = ["left_knee_joint"]
        right_knee_joints = ["right_knee_joint"]
        left_arm_joints = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                           "left_shoulder_yaw_joint", "left_elbow_joint",
                           "left_wrist_roll_joint", "left_wrist_pitch_joint",
                           "left_wrist_yaw_joint"]
        right_arm_joints = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                            "right_shoulder_yaw_joint", "right_elbow_joint",
                            "right_wrist_roll_joint", "right_wrist_pitch_joint",
                            "right_wrist_yaw_joint"]
        waist_joints = ["waist_yaw_joint"]
        knee_joints = ["left_knee_joint", "right_knee_joint"]
        ankle_joints = ["left_ankle_pitch_joint", "left_ankle_roll_joint",
                        "right_ankle_pitch_joint", "right_ankle_roll_joint"]
        # A3 URDF keeps real head links (head welded): track head height there.
        # HoST's G1 URDF instead adds marker links named keyframe/keyframe_head —
        # if the env class hard-requires them, add fixed marker links to the
        # URDF (README step 3b) rather than editing the env.
        keyframe_name = "keyframe"
        head_name = "head_pitch"
        trunk_names = ["pelvis", "torso"]
        base_name = "pelvis_link"
        left_upper_body_names = ["left_shoulder_pitch", "left_elbow"]
        right_upper_body_names = ["right_shoulder_pitch", "right_elbow"]
        left_lower_body_names = ["left_hip_pitch", "left_ankle_roll", "left_knee"]
        right_lower_body_names = ["right_hip_pitch", "right_ankle_roll", "right_knee"]
        left_ankle_names = ["left_ankle_roll"]
        right_ankle_names = ["right_ankle_roll"]
        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.01
        thickness = 0.01
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 1.00          # stand pelvis 1.063 (G1: 0.75)
        only_positive_rewards = False
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 1.30          # ~75% of 1.74 (G1: 1.0)
        target_head_margin = 1
        target_base_height_phase1 = 0.64   # G1 0.45 x (1.063/0.75)
        target_base_height_phase2 = 0.64
        target_base_height_phase3 = 0.92   # G1 0.65 x (1.063/0.75)
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25
        reward_groups = ["task", "regu", "style", "target"]
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1

    class constraints(LeggedRobotCfg.rewards):
        is_gaussian = True
        target_head_height = 1.30
        target_head_margin = 1
        orientation_height_threshold = 0.9
        target_base_height = 0.64
        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        post_task = False

        class scales:
            # Torque penalties rescaled ~1/4: A3 peak torques ~2x G1's, so
            # tau^2 penalties would otherwise dominate 4x harder. First knob
            # to revisit if the policy refuses to use its legs.
            regu_dof_acc = -2.5e-7
            regu_action_rate = -0.01
            regu_smoothness = -0.01
            regu_torques = -6e-7
            regu_joint_power = -6e-6
            regu_dof_vel = -1e-3
            regu_joint_tracking_error = -0.00025
            regu_dof_pos_limits = -100.0
            regu_dof_vel_limits = -1
            style_waist_deviation = -10
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            style_shoulder_roll_deviation = -2.5
            style_left_foot_displacement = 2.5
            style_right_foot_displacement = 2.5
            style_knee_deviation = -0.25
            style_shank_orientation = 10
            style_ground_parallel = 20
            style_feet_distance = -10
            style_style_ang_vel_xy = 1
            target_ang_vel_xy = 10
            target_lin_vel_xy = 10
            target_feet_height_var = 2.5
            target_target_upper_dof_pos = 10
            target_target_orientation = 10
            target_target_base_height = 10

    class domain_rand:
        use_random = True
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]
        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]
        randomize_payload_mass = use_random
        payload_mass_range = [-2, 5]
        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]
        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        randomize_friction = use_random
        friction_range = [0.1, 1]
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.5
        delay = use_random
        max_delay_timesteps = 5

    class curriculum:
        pull_force = True
        force = 175                # x2 trunk links ~ 350 N ~ 60% of 590 N weight
        dof_vel_limit = 300
        base_vel_limit = 20
        threshold_height = 1.2     # head-height curriculum trigger (~70% of 1.74)
        no_orientation = False

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1


class A3UltraCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1   # HoST issue #42: MUST be > 0 or L2C2 no-ops

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        save_interval = 500
        experiment_name = "a3_ultra_ground"
        algorithm_class_name = "PPO"
        init_at_random_ep_len = True
        max_iterations = 12000
