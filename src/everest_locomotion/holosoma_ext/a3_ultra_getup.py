"""Holosoma `--import-file` extension: AgiBot A3 Ultra get-up task (E12/E13).

Trains a policy that stands the robot up from arbitrary fallen poses and ends
in the locomotion policy's expected initial state (manifest `getup.terminal`),
so the get-up -> locomotion handoff is a plain switch.

Recipe: HoST (arXiv:2502.08378) adapted to Holosoma's single-critic PPO —
staged task rewards (upright -> rise -> target pose -> stand still), assist
("pull") force curriculum annealed by success rate, smoothness regularizers
ramped in by the native PenaltyCurriculum, fallen-pose initial states from the
drop-and-settle bank (assets/a3_ultra/getup/, HumanUP method).

Usage (single --import-file; this module pulls in a3_ultra_presets itself):
    python src/holosoma/holosoma/train_agent.py exp:a3-ultra-getup \
        --import-file .../src/everest_locomotion/holosoma_ext/a3_ultra_getup.py
Default simulator is IsaacSim (the only training-validated backend for us).
For local MJWarp smoke runs add `simulator:mjwarp
--simulator.config.sim.max-episode-length-s 10`.

Design notes / holosoma integration facts this file relies on:
- --import-file loads under a synthetic module name; we self-alias into
  sys.modules as "everest_getup" so `env_class` dotted paths and
  "everest_getup:term" func strings resolve.
- Reset order is robot-states callback -> randomization (randomize_dof_state)
  -> command terms. The pose-bank reset is a COMMAND term (WBT MotionCommand
  precedent) so nothing clobbers it.
- Root states are [pos3 | quat XYZW | linvel3 | angvel3]; the pose bank stores
  MuJoCo wxyz quats and MJCF tree-order joints -> remapped here via the bank's
  meta file.
- Reward terms with weight == 0.0 are dropped at manager init; PenaltyCurriculum
  scales terms tagged "penalty_curriculum" using average episode length.
- use_symmetry must stay False: custom obs term names have no mirror in
  SymmetryUtils, and side-lying starts are asymmetric anyway.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

# --- stable module alias so "everest_getup:fn" / "everest_getup.Cls" resolve ---
sys.modules.setdefault("everest_getup", sys.modules[__name__])

# --- pull in the A3 robot + locomotion presets (same directory) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import a3_ultra_presets  # noqa: E402  (registers robot "a3_ultra_29dof")

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg  # noqa: E402
from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg  # noqa: E402
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig  # noqa: E402
from holosoma.config_types.observation import (  # noqa: E402
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg  # noqa: E402
from holosoma.config_types.termination import (  # noqa: E402
    TerminationManagerCfg,
    TerminationTermCfg,
)
from holosoma.config_values import action, algo, simulator, terrain  # noqa: E402
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY  # noqa: E402
from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization  # noqa: E402
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager  # noqa: E402
from holosoma.managers.command.base import CommandTermBase  # noqa: E402
from holosoma.managers.curriculum.base import CurriculumTermBase  # noqa: E402
from holosoma.managers.observation.terms.locomotion import get_projected_gravity  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type  # noqa: E402
from loguru import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Task constants (manifest configs/robots/a3_ultra.yaml `getup:` + default_pose)
# ---------------------------------------------------------------------------
STAND_BASE_HEIGHT = 1.063      # locomotion default_pose pelvis height
HEIGHT_TARGET = 1.00           # dense-shaping target (final pose reward pulls the rest)
LYING_BASE_HEIGHT = 0.15       # reference "flat on ground" pelvis height
STAND_GATE_HEIGHT = 0.80       # gates for final-stage rewards / assist cutoff
UPRIGHT_GATE = -0.85           # projected gravity z when "upright enough"
HOLD_STEPS = 100               # 2 s @ 50 Hz (manifest getup.terminal.hold_time_s)
SUCCESS_HEIGHT = 0.90
MAX_ASSIST_FORCE_N = 350.0     # ~60% of 590 N weight (HoST heuristic)
WRIST_ACTION_FACTOR = 0.2      # 0.25 * 0.2 = manifest wrist_action_scale 0.05
POSES_DIR_DEFAULT = os.path.normpath(os.path.join(a3_ultra_presets.ASSET_ROOT, "..", "getup"))
POSES_DIR = os.environ.get("EVEREST_A3_GETUP_POSES", POSES_DIR_DEFAULT)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class A3UltraGetupManager(LeggedRobotLocomotionManager):
    """Get-up task env: fallen-pose starts, assist force, success tracking."""

    def _get_task_name(self) -> str:
        return "getup"

    # -- buffers ------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        self.getup_assist_scale = 1.0  # annealed by GetupAssistCurriculum
        self._assist_was_zero = False
        self._stand_hold_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self._ever_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._success_rate = torch.tensor(0.0, device=self.device)
        self._getup_prev_dof_vel = torch.zeros(n, self.num_dof, device=self.device)
        self.getup_dof_acc = torch.zeros(n, self.num_dof, device=self.device)
        self._wrist_dof_idx = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "wrist" in name],
            dtype=torch.long, device=self.device,
        )
        self._arm_dof_mask = torch.zeros(self.num_dof, device=self.device)
        for i, name in enumerate(self.dof_names):
            if any(k in name for k in ("shoulder", "elbow", "wrist")):
                self._arm_dof_mask[i] = 1.0
        self._torso_body_idx = self.simulator.find_rigid_body_indice(
            self.robot_config.torso_name
        )

    # -- per-step task state (runs before termination/reward on all backends) --
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        dt = self.dt
        self.getup_dof_acc[:] = (self.simulator.dof_vel - self._getup_prev_dof_vel) / dt
        self._getup_prev_dof_vel[:] = self.simulator.dof_vel

        heights = self.terrain_manager.get_state("locomotion_terrain").base_heights
        pg_z = get_projected_gravity(self)[:, 2]
        lin_vel = torch.norm(self.simulator.robot_root_states[:, 7:10], dim=-1)
        ang_vel = torch.norm(self.simulator.robot_root_states[:, 10:13], dim=-1)
        standing = (
            (heights > SUCCESS_HEIGHT)
            & (pg_z < UPRIGHT_GATE)
            & (lin_vel < 0.4)
            & (ang_vel < 0.8)
        )
        self._stand_hold_count = torch.where(
            standing, self._stand_hold_count + 1, torch.zeros_like(self._stand_hold_count)
        )
        self._ever_success |= self._stand_hold_count >= HOLD_STEPS
        self.log_dict["getup_success_rate"] = self._success_rate.detach().cpu()
        self.log_dict["getup_assist_scale"] = torch.tensor(float(self.getup_assist_scale))

    # -- success EMA + counter reset on episode end ---------------------------
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if len(env_ids) > 0 and target_buf is None:
            batch = self._ever_success[env_ids].float().mean()
            weight = min(float(len(env_ids)) / max(1.0, 0.25 * self.num_envs), 1.0) * 0.2
            self._success_rate = self._success_rate * (1 - weight) + batch * weight
            self._ever_success[env_ids] = False
            self._stand_hold_count[env_ids] = 0
            self._getup_prev_dof_vel[env_ids] = 0.0
        super()._reset_buffers_callback(env_ids, target_buf)

    @property
    def getup_success_rate(self) -> float:
        return float(self._success_rate.detach().cpu().item())

    @property
    def getup_easy_start_prob(self) -> float:
        # debug override: force all-standing starts to isolate task machinery
        # from lying-pose contact physics (MJWarp NaN triage)
        if os.environ.get("EVEREST_GETUP_FORCE_EASY"):
            return 1.0
        # more easy (standing) starts while assist is strong; floor keeps the
        # stand-still stage in the data forever
        return 0.05 + 0.25 * float(self.getup_assist_scale)

    # -- wrist soft-freeze (manifest getup.wrist_action_scale) ---------------
    def _pre_physics_step(self, actions):
        if self._wrist_dof_idx.numel() > 0:
            actions = actions.clone()
            actions[:, self._wrist_dof_idx] *= WRIST_ACTION_FACTOR
        super()._pre_physics_step(actions)

    # -- assist ("pull") force ------------------------------------------------
    def _apply_force_in_physics_step(self):
        super()._apply_force_in_physics_step()
        self._apply_assist_force()

    def _apply_assist_force(self):
        if os.environ.get("EVEREST_GETUP_DISABLE_ASSIST"):
            return
        scale = float(self.getup_assist_scale)
        if scale <= 0.0:
            if not self._assist_was_zero:
                self._clear_assist_force()
                self._assist_was_zero = True
            return
        self._assist_was_zero = False

        heights = self.terrain_manager.get_state("locomotion_terrain").base_heights
        # help only while below the stand gate — assist must never hold the stand
        mask = (heights < STAND_GATE_HEIGHT).float()
        fz = mask * (scale * MAX_ASSIST_FORCE_N)  # [N]

        simtype = get_simulator_type()
        if simtype is SimulatorType.ISAACGYM:
            from isaacgym import gymapi, gymtorch

            force_tensor = torch.zeros(
                self.num_envs, self.simulator.bodies_per_env, 3, device=self.device
            )
            force_tensor[:, self._torso_body_idx, 2] = fz
            self.simulator.gym.apply_rigid_body_force_tensors(
                self.simulator.sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.ENV_SPACE
            )
        elif simtype is SimulatorType.ISAACSIM:
            from isaaclab.utils.math import quat_apply_inverse

            isaac_body_id = self.simulator.body_ids[self._torso_body_idx]
            force_w = torch.zeros(self.num_envs, 3, device=self.simulator.sim_device)
            force_w[:, 2] = fz
            # IsaacLab applies external wrenches in the BODY frame (is_global=False)
            body_quat_w = self.simulator._robot.data.body_quat_w[:, isaac_body_id]
            force_b = quat_apply_inverse(body_quat_w, force_w).unsqueeze(1)  # [N,1,3]
            self.simulator._robot.set_external_force_and_torque(
                forces=force_b,
                torques=torch.zeros_like(force_b),
                body_ids=torch.tensor([isaac_body_id], device=self.simulator.sim_device),
            )
        elif simtype is SimulatorType.MUJOCO:
            mj_body_id = self.simulator.body_ids[self._torso_body_idx]
            applied = self.simulator.applied_forces
            if isinstance(applied, torch.Tensor):  # WarpBackend [N, nbody, 6]
                applied[:, mj_body_id, :3] = 0.0
                applied[:, mj_body_id, 2] = fz
            else:  # ClassicBackend, single env, numpy [nbody, 6]
                applied[mj_body_id, :3] = 0.0
                applied[mj_body_id, 2] = float(fz[0])
        else:  # pragma: no cover
            logger.warning(f"assist force unsupported on simulator {simtype}; disabling")
            self.getup_assist_scale = 0.0

    def _clear_assist_force(self):
        simtype = get_simulator_type()
        if simtype is SimulatorType.ISAACSIM:
            isaac_body_id = self.simulator.body_ids[self._torso_body_idx]
            zeros = torch.zeros(self.num_envs, 1, 3, device=self.simulator.sim_device)
            self.simulator._robot.set_external_force_and_torque(
                forces=zeros, torques=zeros.clone(),
                body_ids=torch.tensor([isaac_body_id], device=self.simulator.sim_device),
            )
        elif simtype is SimulatorType.MUJOCO:
            mj_body_id = self.simulator.body_ids[self._torso_body_idx]
            applied = self.simulator.applied_forces
            if isinstance(applied, torch.Tensor):
                applied[:, mj_body_id, :] = 0.0
            else:
                applied[mj_body_id, :] = 0.0
        # IsaacGym: forces are per-step impulses, nothing persists

    # -- no velocity commands in this task ------------------------------------
    def set_is_evaluating(self, command=None):
        # skip LeggedRobotLocomotionManager's version (touches command tensors)
        logger.info("Setting getup env to evaluating")
        from holosoma.envs.base_task.base_task import BaseTask

        BaseTask.set_is_evaluating(self)

    def _setup_simulator_control(self):
        pass

    # -- checkpointing ---------------------------------------------------------
    def get_checkpoint_state(self):
        state = super().get_checkpoint_state()
        state["getup_assist_scale"] = float(self.getup_assist_scale)
        state["getup_success_rate"] = self._success_rate.detach().cpu()
        return state

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        if not state:
            return
        if "getup_assist_scale" in state:
            self.getup_assist_scale = float(state["getup_assist_scale"])
        if "getup_success_rate" in state:
            self._success_rate = torch.as_tensor(
                float(state["getup_success_rate"]), device=self.device
            )


# ---------------------------------------------------------------------------
# Command term: fallen-pose initial states (runs AFTER randomize_dof_state)
# ---------------------------------------------------------------------------
class PoseBankCommand(CommandTermBase):
    """Reset envs to random settled fallen poses from the .npy bank.

    A fraction of envs (env.getup_easy_start_prob, annealed with the assist
    curriculum) instead starts standing at the locomotion default pose so the
    final stabilize-and-hold stage is always present in the data.
    """

    def setup(self) -> None:
        env = self.env
        params = self.cfg.params or {}
        poses_dir = Path(params.get("poses_dir", POSES_DIR))
        meta = json.loads((poses_dir / "fallen_poses_meta.json").read_text())
        tree_names = meta["hinge_joint_names_tree_order"]
        perm = [tree_names.index(n) for n in env.dof_names]

        import numpy as np

        banks = []
        for cat in ("supine", "prone", "side_left", "side_right"):
            arr = np.load(poses_dir / f"fallen_poses_{cat}.npy")
            if len(arr):
                banks.append(arr)
        bank = np.concatenate(banks, axis=0)
        if len(bank) < 100:
            raise RuntimeError(f"fallen pose bank too small ({len(bank)}) in {poses_dir}")

        root_pos = torch.tensor(bank[:, 0:3], dtype=torch.float, device=env.device)
        quat_wxyz = torch.tensor(bank[:, 3:7], dtype=torch.float, device=env.device)
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        dof_pos = torch.tensor(bank[:, 7:], dtype=torch.float, device=env.device)[:, perm]

        self.bank_root_z = root_pos[:, 2].contiguous()
        self.bank_quat = quat_xyzw.contiguous()
        self.bank_dof_pos = dof_pos.contiguous()
        self.joint_noise = float(params.get("joint_noise_rad", 0.05))
        self.num_poses = len(bank)
        logger.info(f"getup pose bank: {self.num_poses} poses from {poses_dir}")

    def reset(self, env_ids) -> None:
        env = self.env
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        origins = env.terrain_manager.get_state("locomotion_terrain").env_origins[env_ids]

        easy = torch.rand(n, device=env.device) < float(env.getup_easy_start_prob)
        rows = torch.randint(0, self.num_poses, (n,), device=env.device)

        # fallen states from the bank
        dof_pos = self.bank_dof_pos[rows]
        noise = (torch.rand_like(dof_pos) * 2 - 1) * self.joint_noise
        limits = env.dof_pos_limits.to(env.device)
        dof_pos = torch.clamp(dof_pos + noise, limits[:, 0], limits[:, 1])
        root_z = self.bank_root_z[rows]
        quat = self.bank_quat[rows]

        # easy starts: locomotion default stand
        stand_quat = torch.zeros(n, 4, device=env.device)
        stand_quat[:, 3] = 1.0
        dof_pos = torch.where(easy.unsqueeze(1), env.default_dof_pos[env_ids], dof_pos)
        quat = torch.where(easy.unsqueeze(1), stand_quat, quat)
        root_z = torch.where(easy, torch.full_like(root_z, STAND_BASE_HEIGHT + 0.005), root_z)

        env.simulator.dof_pos[env_ids] = dof_pos
        env.simulator.dof_vel[env_ids] = 0.0
        rs = env.simulator.robot_root_states
        rs[env_ids, 0:2] = origins[:, 0:2]
        rs[env_ids, 2] = origins[:, 2] + root_z
        rs[env_ids, 3:7] = quat
        rs[env_ids, 7:13] = 0.0

    def step(self) -> None:
        return


# ---------------------------------------------------------------------------
# Curriculum term: anneal assist force by success rate
# ---------------------------------------------------------------------------
class GetupAssistCurriculum(CurriculumTermBase):
    """HoST pull-force curriculum, driven by held-stand success rate.

    Every `interval_resets` env-episode ends: success above `up_threshold`
    lowers the assist scale by `step`; below `down_threshold` raises it.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params or {}
        self.step_size = float(p.get("step", 0.05))
        self.up_threshold = float(p.get("up_threshold", 0.7))
        self.down_threshold = float(p.get("down_threshold", 0.3))
        self.min_scale = float(p.get("min_scale", 0.0))
        self.max_scale = float(p.get("max_scale", 1.0))
        self.interval_resets = int(p.get("interval_resets", 4096))
        self._resets_seen = 0

    def setup(self) -> None:
        return

    def reset(self, env_ids) -> None:
        n = env_ids.numel() if torch.is_tensor(env_ids) else len(env_ids)
        if n == 0:
            return
        self._resets_seen += n
        if self._resets_seen < self.interval_resets:
            return
        self._resets_seen = 0
        rate = self.env.getup_success_rate
        scale = float(self.env.getup_assist_scale)
        if rate > self.up_threshold:
            scale -= self.step_size
        elif rate < self.down_threshold:
            scale += self.step_size
        self.env.getup_assist_scale = min(self.max_scale, max(self.min_scale, scale))

    def step(self) -> None:
        return


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------
def task_upright(env) -> torch.Tensor:
    """0 when inverted, 1 when upright (dense righting signal)."""
    pg_z = get_projected_gravity(env)[:, 2]
    return (1.0 - pg_z) * 0.5


def task_base_height(env, target: float = HEIGHT_TARGET, floor: float = LYING_BASE_HEIGHT) -> torch.Tensor:
    """0 lying -> 1 at target pelvis height (dense rising signal)."""
    h = env.terrain_manager.get_state("locomotion_terrain").base_heights
    return torch.clamp((h - floor) / (target - floor), 0.0, 1.0)


def _stand_gate(env) -> torch.Tensor:
    h = env.terrain_manager.get_state("locomotion_terrain").base_heights
    pg_z = get_projected_gravity(env)[:, 2]
    return ((h > STAND_GATE_HEIGHT) & (pg_z < UPRIGHT_GATE)).float()


def task_target_pose(env, sigma: float = 2.0, wrist_weight: float = 0.1) -> torch.Tensor:
    """Track the locomotion default pose once roughly standing (handoff pose)."""
    err = env.simulator.dof_pos - env.default_dof_pos
    w = torch.ones(env.num_dof, device=env.device)
    for i, name in enumerate(env.dof_names):
        if "wrist" in name:
            w[i] = wrist_weight
    return torch.exp(-torch.sum(w * err * err, dim=1) / sigma) * _stand_gate(env)


def task_stand_still(env) -> torch.Tensor:
    """Zero base velocity once standing (manifest terminal velocity tolerances)."""
    lin = torch.sum(env.simulator.robot_root_states[:, 7:10] ** 2, dim=1)
    ang = torch.sum(env.simulator.robot_root_states[:, 10:13] ** 2, dim=1)
    return torch.exp(-(lin + 0.2 * ang)) * _stand_gate(env)


def task_feet_contact(env) -> torch.Tensor:
    """Both feet loaded once standing (clean double-support stance)."""
    contact = (env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0).float()
    return contact.mean(dim=1) * _stand_gate(env)


def penalty_dof_vel(env) -> torch.Tensor:
    return torch.sum(env.simulator.dof_vel**2, dim=1)


def penalty_dof_acc(env) -> torch.Tensor:
    return torch.sum(env.getup_dof_acc**2, dim=1)


def penalty_torques(env, arm_weight: float = 3.0) -> torch.Tensor:
    """Normalized torque penalty, arms weighted (24 Nm elbows on a 60 kg body)."""
    tau = env.action_manager.get_term("joint_control").torques
    limits = env.torque_limits.to(env.device).flatten()
    w = 1.0 + (arm_weight - 1.0) * env._arm_dof_mask
    return torch.sum(w * (tau / limits) ** 2, dim=1)


# ---------------------------------------------------------------------------
# Termination terms (existing dof-limit terms are dead in manager envs)
# ---------------------------------------------------------------------------
def getup_dof_velocity_exceeded(env, tolerance: float = 1.25) -> torch.Tensor:
    limits = env.dof_vel_limits.to(env.device).flatten()
    return torch.any(torch.abs(env.simulator.dof_vel) > tolerance * limits, dim=1)


def getup_base_velocity_exceeded(env, max_lin_vel: float = 4.0) -> torch.Tensor:
    # non-finite catch is load-bearing: NaN states never trip a ">" comparison,
    # so without it a single blown env poisons the whole PPO update
    rs = env.simulator.robot_root_states
    state = rs[:, 0:13]  # slicing, not the raw view: MJWarp's MjwRootStateView is not a Tensor
    blown = torch.norm(state[:, 7:10], dim=-1) > max_lin_vel
    dof_vel = env.simulator.dof_vel[:, :]
    return blown | ~torch.isfinite(state).all(dim=1) | ~torch.isfinite(dof_vel).all(dim=1)


# ---------------------------------------------------------------------------
# Observation term (critic-only privileged height)
# ---------------------------------------------------------------------------
def obs_base_height(env) -> torch.Tensor:
    return env.terrain_manager.get_state("locomotion_terrain").base_heights.unsqueeze(1)


# ---------------------------------------------------------------------------
# Manager configs
# ---------------------------------------------------------------------------
_L = "holosoma.managers.observation.terms.locomotion"
_actor_terms = {
    "base_ang_vel": ObsTermCfg(func=f"{_L}:base_ang_vel", scale=0.25, noise=0.0),
    "projected_gravity": ObsTermCfg(func=f"{_L}:projected_gravity", scale=1.0, noise=0.0),
    "dof_pos": ObsTermCfg(func=f"{_L}:dof_pos", scale=1.0, noise=0.01),
    "dof_vel": ObsTermCfg(func=f"{_L}:dof_vel", scale=0.05, noise=0.1),
    "actions": ObsTermCfg(func=f"{_L}:actions", scale=1.0, noise=0.0),
}
getup_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=1, terms=dict(_actor_terms)
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                **{k: replace(v, noise=0.0) for k, v in _actor_terms.items()},
                "base_lin_vel": ObsTermCfg(func=f"{_L}:base_lin_vel", scale=2.0, noise=0.0),
                "base_height": ObsTermCfg(func="everest_getup:obs_base_height", scale=1.0, noise=0.0),
            },
        ),
    }
)

getup_reward = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # -- staged task rewards (never curriculum-scaled) --
        "task_upright": RewardTermCfg(func="everest_getup:task_upright", weight=1.0, params={}),
        "task_base_height": RewardTermCfg(func="everest_getup:task_base_height", weight=2.0, params={}),
        "task_target_pose": RewardTermCfg(func="everest_getup:task_target_pose", weight=2.5, params={}),
        "task_stand_still": RewardTermCfg(func="everest_getup:task_stand_still", weight=1.5, params={}),
        "task_feet_contact": RewardTermCfg(func="everest_getup:task_feet_contact", weight=0.5, params={}),
        # -- smoothness / hardware-protection (ramped in by PenaltyCurriculum) --
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-1.0, params={}, tags=["penalty_curriculum"],
        ),
        "penalty_dof_vel": RewardTermCfg(
            func="everest_getup:penalty_dof_vel", weight=-5e-3, params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_dof_acc": RewardTermCfg(
            func="everest_getup:penalty_dof_acc", weight=-1e-6, params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_torques": RewardTermCfg(
            func="everest_getup:penalty_torques", weight=-0.2, params={"arm_weight": 3.0},
            tags=["penalty_curriculum"],
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-10.0, params={"soft_dof_pos_limit": 0.95},
        ),
    },
)

getup_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        # hard speed limits double as anti-flail safety (HoST) — NO contact
        # termination: ground contact is the whole point of this task
        "dof_velocity": TerminationTermCfg(
            func="everest_getup:getup_dof_velocity_exceeded", params={"tolerance": 1.25}
        ),
        "base_velocity": TerminationTermCfg(
            func="everest_getup:getup_base_velocity_exceeded", params={"max_lin_vel": 4.0}
        ),
    }
)

getup_command = CommandManagerCfg(
    params={},
    setup_terms={
        "pose_bank": CommandTermCfg(func="everest_getup:PoseBankCommand", params={}),
    },
    reset_terms={
        "pose_bank": CommandTermCfg(func="everest_getup:PoseBankCommand"),
    },
    step_terms={},
)

getup_curriculum = CurriculumManagerCfg(
    params={"num_compute_average_epl": 1000},
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        # episodes end early only on velocity blowups, so average episode
        # length is a smoothness proxy: ramp penalties in as flailing stops
        # (max episode = 10 s @ 50 Hz = 500 steps)
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.1,
                "min_scale": 0.1,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 400.0,
                "degree": 0.001,
            },
        ),
        "getup_assist": CurriculumTermCfg(
            func="everest_getup:GetupAssistCurriculum",
            params={
                "step": 0.05,
                "up_threshold": 0.7,
                "down_threshold": 0.3,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "interval_resets": 4096,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

# DR: keep the locomotion recipe minus pushes (velocity-impulse pushes make no
# sense mid-ground-contact) and minus randomize_dof_state (would fight the
# pose bank; the bank + joint noise is the initial-state randomization).
_PUSH_KEYS = {"push_randomizer_state", "randomize_push_schedule", "apply_pushes"}
getup_randomization = replace(
    g1_29dof_randomization,
    setup_terms={k: v for k, v in g1_29dof_randomization.setup_terms.items() if k not in _PUSH_KEYS},
    reset_terms={
        k: v
        for k, v in g1_29dof_randomization.reset_terms.items()
        if k not in _PUSH_KEYS and k != "randomize_dof_state"
    },
    step_terms={k: v for k, v in g1_29dof_randomization.step_terms.items() if k not in _PUSH_KEYS},
)

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
_getup_sim = replace(
    simulator.isaacsim,
    config=replace(
        simulator.isaacsim.config,
        sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
    ),
)

_getup_experiment_base = ExperimentConfig(
        env_class="everest_getup.A3UltraGetupManager",
        training=TrainingConfig(project="everest-a3", name="a3_ultra_getup"),
        algo=replace(
            algo.ppo,
            config=replace(
                algo.ppo.config,
                num_learning_iterations=20000,
                use_symmetry=False,  # custom obs terms have no mirrors; side starts are asymmetric
            ),
        ),
        simulator=_getup_sim,
        # lying/rolling makes far more simultaneous contacts than locomotion:
        # 64-env MJWarp smoke needed njmax ~670 (overflowed at the loco value 16)
        robot=replace(a3_ultra_presets.a3_ultra_29dof, contact_pairs_multiplier=48),
        terrain=terrain.terrain_locomotion_plane,
        observation=getup_observation,
        action=action.g1_29dof_joint_pos,
        termination=getup_termination,
        randomization=getup_randomization,
        command=getup_command,
        curriculum=getup_curriculum,
        reward=getup_reward,
)

a3_ultra_getup = EXPERIMENT_REGISTRY.add("a3_ultra_getup", _getup_experiment_base)

# FastSAC variant: the repo's proven algo (E00/E01) and the only one that
# survives MJWarp locally — use for local smoke runs and as an algo ablation
# on the cloud (PPO is the HoST-parity default).
a3_ultra_getup_fast_sac = EXPERIMENT_REGISTRY.add(
    "a3_ultra_getup_fast_sac",
    replace(
        _getup_experiment_base,
        training=TrainingConfig(project="everest-a3", name="a3_ultra_getup_fast_sac"),
        algo=replace(
            algo.fast_sac,
            config=replace(
                algo.fast_sac.config,
                num_learning_iterations=50000,
                use_symmetry=False,
            ),
        ),
    ),
)
