"""Holosoma `--import-file` extension: A3 Ultra **final walking policy** (v2).

Implements `docs/final_rl_policy.md` — components **A–G** plus the §4 extras —
as a staged ladder S0→S4 of registered experiments. Locomotion only; the
get-up track (`a3_ultra_getup.py`) is untouched.

    A  heading command            -> fixes yaw drift
    B  concurrent velocity estimator -> fixes the ~15% speed undershoot
    C  upper-body pose curriculum -> arms become a trained-against disturbance
    D  centroidal angular momentum rewards -> arms start helping
    E  height scan (13x9 scandots) -> terrain perception
    F  smoothness stack (jerk / dof-acc / torque, + optional Lipschitz penalty)
    G  observation history (1 -> 5)

Usage (single --import-file; this module pulls in a3_ultra_presets itself):

    python src/holosoma/holosoma/train_agent.py exp:a3-ultra-loco-v2-s1 \
        --import-file .../src/everest_locomotion/holosoma_ext/a3_ultra_loco_v2.py \
        --training.num-envs 4096

Stages (see `scripts/cloud/train_a3_v2_cloud.sh`):

    a3-ultra-loco-v2-s0        every feature OFF -- refactor-neutrality control
    a3-ultra-loco-v2-s1        + A heading, B estimator, G history, wider command
    a3-ultra-loco-v2-s2        + C arm curriculum, D CAM, arm pose weights -> 0
    a3-ultra-loco-v2-s3        + E height scan, slope terrain
    a3-ultra-loco-v2-s4        + F smoothness reward terms
    a3-ultra-loco-v2-s4-lcp    + F as a Lipschitz gradient penalty (ablation)

Holosoma integration facts this file relies on (all verified against `6e146b0`,
not inferred):

- `--import-file` loads under a synthetic module name; we self-alias into
  `sys.modules` as ``everest_loco_v2`` so ``env_class`` dotted paths,
  ``"everest_loco_v2:term"`` func strings and ``algo._target_`` resolve.
- **No Holosoma fork is needed.** ``FastSACAlgoConfig._target_`` is a plain
  dotted path resolved by ``train_agent.get_class``, so component **B** ships as
  a `FastSACAgent` subclass *here*. (`docs/final_rl_policy.md` §3B assumed a
  fork; it is not required — see `notes/decisions.md`.)
- Observation groups concatenate their terms in **alphabetically sorted** order
  (`ObservationManager.compute_group`), and history is stored **per term**:
  a group is ``[term_a(t0..tT) | term_b(t0..tT) | ...]``, oldest frame first.
- ``SymmetryUtils.mirror_xz_plane`` assumes the *opposite* (frame-major) layout
  and is therefore **wrong whenever history_length > 1**. We patch it below;
  without the patch, G + `use_symmetry=True` silently mirrors garbage.
- Reward terms with weight ``0.0`` are dropped at manager init; `PenaltyCurriculum`
  scales terms tagged ``"penalty_curriculum"`` by average episode length.
- ``env.action_scales`` is the per-DOF action scale (0.25 uniform here), and the
  PD target is ``default_dof_pos + action_scales * action`` — that is the mapping
  the arm-override in `_pre_physics_step` inverts.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from typing import Any, Sequence

# --- stable module alias so "everest_loco_v2:fn" / "everest_loco_v2.Cls" resolve ---
sys.modules.setdefault("everest_loco_v2", sys.modules[__name__])

# --- pull in the A3 robot + v1 locomotion presets (same directory) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import a3_ultra_presets  # noqa: E402  (registers robot "a3_ultra_29dof")

from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor  # noqa: E402
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent  # noqa: E402
from holosoma.agents.modules.augmentation_utils import SymmetryUtils  # noqa: E402
from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg  # noqa: E402
from holosoma.config_types.curriculum import (  # noqa: E402
    CurriculumManagerCfg,
    CurriculumTermCfg,
)
from holosoma.config_types.experiment import (  # noqa: E402
    ExperimentConfig,
    NightlyConfig,
    TrainingConfig,
)
from holosoma.config_types.observation import (  # noqa: E402
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg  # noqa: E402
from holosoma.config_values import action, algo, simulator, terrain  # noqa: E402
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY  # noqa: E402
from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization  # noqa: E402
from holosoma.config_values.loco.g1.reward import g1_29dof_loco_fast_sac  # noqa: E402
from holosoma.config_values.loco.g1.termination import g1_29dof_termination  # noqa: E402
from holosoma.envs.locomotion.locomotion_manager import (  # noqa: E402
    LeggedRobotLocomotionManager,
)
from holosoma.managers.command.base import CommandTermBase  # noqa: E402
from holosoma.managers.command.terms.locomotion import LocomotionCommand  # noqa: E402
from holosoma.managers.curriculum.base import CurriculumTermBase  # noqa: E402
from holosoma.managers.observation.terms.locomotion import base_forward_vector  # noqa: E402
from holosoma.utils.rotations import quat_apply, quat_apply_yaw, wrap_to_pi  # noqa: E402
from holosoma.utils.safe_torch_import import F, nn, optim, torch  # noqa: E402
from holosoma.utils.torch_utils import torch_rand_float  # noqa: E402
from loguru import logger  # noqa: E402

_L = "holosoma.managers.observation.terms.locomotion"
_R = "holosoma.managers.reward.terms.locomotion"
_C = "holosoma.managers.curriculum.terms.locomotion"
_X = "everest_loco_v2"

# ---------------------------------------------------------------------------
# Robot-derived constants (canonical DOF order = a3_ultra_presets.DOF_NAMES)
# ---------------------------------------------------------------------------
DOF_NAMES = list(a3_ultra_presets.DOF_NAMES)
NUM_DOF = len(DOF_NAMES)  # 29
ARM_DOF_IDX = [
    i for i, n in enumerate(DOF_NAMES) if any(k in n for k in ("shoulder", "elbow", "wrist"))
]
NUM_ARM_DOF = len(ARM_DOF_IDX)  # 14

# Height scan geometry (component E). 13 x 9 == FastSACConfig.encoder_obs_shape
# default (1, 13, 9), so CNNActor/CNNCritic need no reshaping changes.
SCAN_NX, SCAN_NY = 13, 9
SCAN_SPACING = 0.1          # m, inside the 6-8 cm band the elevation-map work converged on
SCAN_X_OFFSET = 0.2         # m ahead of the base
SCAN_NOMINAL_HEIGHT = 1.05  # m: pelvis height that makes flat ground read ~0
SCAN_CLIP = 1.0             # m
SCAN_DIM = SCAN_NX * SCAN_NY  # 117

# Velocity estimator (component B)
VEL_EST_DIM = 3
VEL_EST_HIDDEN = (256, 128)
VEL_EST_LR = 1e-3
VEL_EST_TARGET_TERM = "base_lin_vel"  # privileged critic term regressed against

# Lipschitz-constrained policy (component F, S4 ablation only)
LCP_LAMBDA = 0.002

# CAM reference amplitude (component D). ASSUMED — no reference motion exists for
# the A3, so the vertical-CAM target is derived from the gait clock and scaled by
# commanded forward speed. Tune here, and log `cam_z` / `cam_xy` before trusting it.
CAM_REF_AMPLITUDE = 0.05  # (m^2/s) per (m/s) of commanded forward speed

# v1's tracking baseline, for the nightly gate on S1+
V1_TRACK_LIN, V1_TRACK_ANG = 0.95, 0.8


def _by_name(env, values: Sequence[float]) -> torch.Tensor:
    """Map a config-ordered per-DOF list into *this backend's* DOF order.

    `env.dof_pos_limits` / `torque_limits` are filled POSITIONALLY from the config
    lists by every backend, but the MuJoCo backend orders its DOF arrays in MJCF
    tree order (waist-first on the A3) while the config lists are canonical
    (legs-first) — so those tensors are scrambled there. IsaacSim resolves by name
    and is unaffected. Never consume them; build tensors with this instead.
    """
    order = [DOF_NAMES.index(n) for n in env.dof_names]
    return torch.tensor([values[j] for j in order], dtype=torch.float, device=env.device)


# ===========================================================================
# Symmetry: correct the history layout, and register mirrors for the new terms
# ===========================================================================
def _mirror_xz_plane_term_major(self, observation, env, obs_list):  # noqa: ARG001
    """Drop-in replacement for `SymmetryUtils.mirror_xz_plane`.

    Upstream reshapes a whole group to ``[B, history, single_frame_dim]``, which
    assumes frame-major storage. `ObservationManager._apply_history` actually
    emits ``[num_envs, history * term_dim]`` **per term** and then concatenates
    terms, i.e. term-major. The two agree only when ``history_length == 1`` —
    which is why v1 was fine and why component **G** would have broken symmetry
    augmentation silently. This walks the real layout instead.
    """
    mirrored_all = observation.clone()
    batch = observation.shape[0]
    idx = 0
    for obs_key in obs_list:
        hist = self.history_lengths[obs_key]
        for sub_key in self.sub_observation_keys[obs_key]:
            dim = self.sub_observation_dims[sub_key]
            width = dim * hist
            block = mirrored_all[..., idx : idx + width].reshape(batch, hist, dim)
            block = getattr(self, f"mirror_obs_{sub_key}")(block)
            mirrored_all[..., idx : idx + width] = block.reshape(batch, width)
            idx += width
    return mirrored_all


SymmetryUtils.mirror_xz_plane = _mirror_xz_plane_term_major


def _mirror_obs_heading_error(self, heading_error):  # noqa: ARG001
    """Yaw error flips sign under an x-z plane mirror."""
    return -heading_error


def _arm_symmetry_maps():
    """(index map, sign mask) for the 14-dim arm block, from the robot config.

    Same left/right swap + sign flips `mirror_action_xz_plane` applies, restricted
    to arm indices and re-based to 0..13.
    """
    cfg = a3_ultra_presets.a3_ultra_29dof
    name_to_local = {DOF_NAMES[g]: k for k, g in enumerate(ARM_DOF_IDX)}
    idx_map = list(range(NUM_ARM_DOF))
    for a, b in cfg.symmetry_joint_names.items():
        if a in name_to_local and b in name_to_local:
            idx_map[name_to_local[a]] = name_to_local[b]
    signs = [
        -1.0 if DOF_NAMES[g] in set(cfg.flip_sign_joint_names) else 1.0 for g in ARM_DOF_IDX
    ]
    return idx_map, signs


_ARM_MIRROR_IDX, _ARM_MIRROR_SIGN = _arm_symmetry_maps()


def _mirror_obs_upper_body_target(self, target):
    cache = getattr(self, "_a3_arm_mirror", None)
    if cache is None:
        cache = (
            torch.tensor(_ARM_MIRROR_IDX, dtype=torch.long, device=target.device),
            torch.tensor(_ARM_MIRROR_SIGN, dtype=torch.float, device=target.device),
        )
        self._a3_arm_mirror = cache
    idx, sign = cache
    return target[..., idx] * sign


def _mirror_obs_height_scan(self, scan):  # noqa: ARG001
    """Mirror the scandot grid across the robot's own x-z plane: reverse y."""
    shape = scan.shape
    grid = scan.reshape(*shape[:-1], SCAN_NX, SCAN_NY)
    return grid.flip(-1).reshape(shape)


SymmetryUtils.mirror_obs_heading_error = _mirror_obs_heading_error
SymmetryUtils.mirror_obs_upper_body_target = _mirror_obs_upper_body_target
SymmetryUtils.mirror_obs_height_scan = _mirror_obs_height_scan


# ===========================================================================
# Environment
# ===========================================================================
class A3UltraLocoV2Manager(LeggedRobotLocomotionManager):
    """Locomotion env with the v2 per-step task state.

    Everything here is *opt-in by config*: buffers are cheap and always
    allocated, but the arm override only fires when an ``upper_body_command``
    term is registered, CAM only when a CAM reward term is active, and the
    height scan only when a `height_scan` observation term asks for it. S0
    therefore behaves exactly like v1's `LeggedRobotLocomotionManager`.
    """

    # -- buffers ------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        n, d = self.num_envs, self.num_dof
        dev = self.device

        # curriculum state must survive reset_all()'s re-run of _init_buffers
        if not hasattr(self, "arm_amplitude"):
            self.arm_amplitude = 0.25  # widened by ArmAmplitudeCurriculum

        self._v2_state_step = -1
        self._prev_dof_vel = torch.zeros(n, d, device=dev)
        self.dof_acc = torch.zeros(n, d, device=dev)
        self._action_tm2 = torch.zeros(n, d, device=dev)

        # feet air time / landing impact
        self.feet_air_time = torch.zeros(n, len(self.feet_indices), device=dev)
        self._air_time_at_landing = torch.zeros(n, len(self.feet_indices), device=dev)
        self.feet_first_contact = torch.zeros(
            n, len(self.feet_indices), dtype=torch.bool, device=dev
        )
        self._last_contacts = torch.zeros(
            n, len(self.feet_indices), dtype=torch.bool, device=dev
        )

        # centroidal angular momentum (world frame, mass-normalised)
        self.cam = torch.zeros(n, 3, device=dev)
        self.cam_rate = torch.zeros(n, 3, device=dev)
        self._body_mass = None  # lazily read (after mass randomisation)

        # per-DOF limits by name (see _by_name)
        cfg = self.robot_config
        self.v2_torque_limits = _by_name(self, cfg.dof_effort_limit_list)
        self.v2_dof_vel_limits = _by_name(self, cfg.dof_vel_limit_list)
        self.v2_pos_limits = torch.stack(
            [
                _by_name(self, cfg.dof_pos_lower_limit_list),
                _by_name(self, cfg.dof_pos_upper_limit_list),
            ],
            dim=1,
        )
        self._arm_idx = torch.tensor(
            [i for i, n_ in enumerate(self.dof_names)
             if any(k in n_ for k in ("shoulder", "elbow", "wrist"))],
            dtype=torch.long, device=dev,
        )
        if self._arm_idx.numel() != NUM_ARM_DOF:
            raise RuntimeError(
                f"expected {NUM_ARM_DOF} arm DOFs, found {self._arm_idx.numel()} in {self.dof_names}"
            )

        # height-scan sample grid, base-local (yaw frame), built once
        xs = torch.linspace(
            -(SCAN_NX - 1) / 2 * SCAN_SPACING, (SCAN_NX - 1) / 2 * SCAN_SPACING, SCAN_NX,
            device=dev,
        ) + SCAN_X_OFFSET
        ys = torch.linspace(
            -(SCAN_NY - 1) / 2 * SCAN_SPACING, (SCAN_NY - 1) / 2 * SCAN_SPACING, SCAN_NY,
            device=dev,
        )
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")  # row-major (x, y) -> (1, 13, 9)
        self._scan_points_local = torch.zeros(SCAN_DIM, 3, device=dev)
        self._scan_points_local[:, 0] = gx.flatten()
        self._scan_points_local[:, 1] = gy.flatten()
        self._scan_cache = None
        self._scan_cache_step = -1

    # -- per-step task state -------------------------------------------------
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        # this callback also fires inside _refresh_envs_after_reset, i.e. twice on
        # any step where an env resets — guard so finite differences and air-time
        # counters advance exactly once per control step
        if self._v2_state_step == self.common_step_counter:
            return
        self._v2_state_step = self.common_step_counter
        dt = self.dt

        self.dof_acc[:] = (self.simulator.dof_vel - self._prev_dof_vel) / dt
        self._prev_dof_vel[:] = self.simulator.dof_vel

        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact
        self.feet_first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += dt
        self._air_time_at_landing = self.feet_air_time.clone()
        self.feet_air_time = self.feet_air_time * (~contact_filt)

        if self._cam_active():
            cam = _compute_cam(self)
            self.cam_rate[:] = (cam - self.cam) / dt
            self.cam[:] = cam
            self.log_dict["cam_z"] = self.cam[:, 2].abs().mean().detach().cpu()
            self.log_dict["cam_xy"] = self.cam[:, :2].norm(dim=1).mean().detach().cpu()

        self.log_dict["arm_amplitude"] = torch.tensor(float(self.arm_amplitude))

    def _cam_active(self) -> bool:
        cached = getattr(self, "_cam_active_cached", None)
        if cached is None:
            manager = getattr(self, "reward_manager", None)
            terms = set(getattr(manager, "active_terms", []) or [])
            cached = bool(terms & {"cam_tracking", "penalty_cam_damping"})
            self._cam_active_cached = cached
        return cached

    # -- arm override (component C) -----------------------------------------
    def _pre_physics_step(self, actions):
        # a_{t-2} for the jerk penalty: the action manager rolls prev<-cur inside
        # process_actions, so *before* super() runs prev_action still holds a_{t-2}
        self._action_tm2[:] = self.action_manager.prev_action

        upper = self.command_manager.get_state("upper_body_command")
        if upper is not None:
            actions = actions.clone()
            target_abs = upper.arm_target  # [N, 14] absolute joint targets (rad)
            scale = self.action_scales[self._arm_idx].unsqueeze(0)
            default = self.default_dof_pos[:, self._arm_idx]
            override = (target_abs - default) / scale
            actions[:, self._arm_idx] = torch.where(
                upper.active.unsqueeze(1), override, actions[:, self._arm_idx]
            )

        super()._pre_physics_step(actions)

    # -- heading command tolerates the 4-dim command vector ------------------
    def set_is_evaluating(self, command=None):
        commands = self.command_manager.commands
        if command is not None and len(command) < commands.shape[1]:
            command = list(command) + [0.0] * (commands.shape[1] - len(command))
        super().set_is_evaluating(command)

    # -- checkpointing -------------------------------------------------------
    def get_checkpoint_state(self):
        state = super().get_checkpoint_state()
        state["arm_amplitude"] = float(self.arm_amplitude)
        return state

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        if state and "arm_amplitude" in state:
            self.arm_amplitude = float(state["arm_amplitude"])


# ===========================================================================
# A · heading command
# ===========================================================================
class HeadingLocomotionCommand(LocomotionCommand):
    """`LocomotionCommand` with entry 3 = target heading, and yaw rate regulated.

    Upstream samples ``command_ranges["heading"]`` and never reads it, so heading
    error integrates freely — the measured 3-6 deg/s drift. This derives the yaw
    command from heading error (the standard `legged_gym` formulation), which
    turns an integrating error into a regulated one.

    ``heading_prob`` of episodes are heading-controlled; the rest keep pure
    yaw-rate commands so spin-in-place does not regress.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.command_dim = 4
        self.heading_prob = float(params.get("heading_prob", 0.8))
        self.heading_gain = float(params.get("heading_gain", 0.5))
        self.heading_clip = float(params.get("heading_clip", 1.0))
        self.heading_mode: torch.Tensor | None = None
        self.stand_mode: torch.Tensor | None = None

    def setup(self) -> None:
        super().setup()
        n, dev = self.env.num_envs, self.env.device
        self.heading_mode = torch.zeros(n, dtype=torch.bool, device=dev)
        self.stand_mode = torch.zeros(n, dtype=torch.bool, device=dev)

    def _resample(self, env_ids: torch.Tensor) -> None:
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return
        dev, r = self.env.device, self.command_ranges
        k = env_ids.shape[0]

        for col, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            commands[env_ids, col] = torch_rand_float(
                r[key][0], r[key][1], (k, 1), device=dev
            ).squeeze(1)
        lo, hi = r.get("heading", (-math.pi, math.pi))
        commands[env_ids, 3] = torch_rand_float(lo, hi, (k, 1), device=dev).squeeze(1)

        manager = getattr(self, "manager", None)
        gait_state = manager.get_state("locomotion_gait") if manager is not None else None
        if gait_state is not None:
            gait_state.resample_frequency(env_ids)

        stand = torch.zeros(k, dtype=torch.bool, device=dev)
        if self.stand_prob > 0.0:
            stand = torch.rand(k, device=dev) <= self.stand_prob
            if stand.any():
                commands[env_ids[stand], :3] = 0.0
        self.stand_mode[env_ids] = stand
        self.heading_mode[env_ids] = (torch.rand(k, device=dev) < self.heading_prob) & ~stand

        self._regulate_yaw()

    def step(self) -> None:
        super().step()
        if not self.env.is_evaluating:
            self._regulate_yaw()

    def _regulate_yaw(self) -> None:
        commands, mode = self.commands, self.heading_mode
        if commands is None or mode is None or not bool(mode.any()):
            return
        omega = torch.clamp(
            self.heading_gain * self.heading_error(), -self.heading_clip, self.heading_clip
        )
        commands[:, 2] = torch.where(mode, omega, commands[:, 2])

    def heading_error(self) -> torch.Tensor:
        """Signed error to the target heading [num_envs]; zero on yaw-rate episodes."""
        err = wrap_to_pi((self.commands[:, 3] - _base_yaw(self.env)).clone())
        return err * self.heading_mode.float()


def _base_quat(env) -> torch.Tensor:
    """`env.base_quat` as a real tensor.

    On the MuJoCo/MJWarp backend it is an `MjwQuaternionView`, which supports
    indexing but not `reshape`/`repeat` — and `quat_apply` reshapes. Slicing
    materialises it on every backend.
    """
    return env.base_quat[:]


def _base_yaw(env) -> torch.Tensor:
    forward = quat_apply(_base_quat(env), base_forward_vector(env), w_last=True)
    return torch.atan2(forward[:, 1], forward[:, 0])


def heading_error(env) -> torch.Tensor:
    """Observation term: signed heading error [num_envs, 1].

    Zero is unambiguous here: a pure yaw-rate episode has no heading target, and
    an on-target heading episode is already doing the right thing — both want the
    same correction (none).
    """
    term = env.command_manager.get_state("locomotion_command")
    if not isinstance(term, HeadingLocomotionCommand) or term.heading_mode is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return term.heading_error().unsqueeze(1)


# ===========================================================================
# C · upper-body pose command + curriculum
# ===========================================================================
class UpperBodyCommand(CommandTermBase):
    """Samples per-env arm targets and a motion type: hold / sinusoid / step.

    Hold and dynamic motion are *different* failure regimes (E08b: sustained
    poses kill v1, oscillation does not), so both must be in the distribution.
    Amplitude is scaled by ``env.arm_amplitude``, widened by
    `ArmAmplitudeCurriculum`. ``active_prob`` of episodes hand the arms to this
    term; the rest leave the policy owning its own arms.
    """

    MODE_HOLD, MODE_SIN, MODE_STEP = 0, 1, 2

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params or {}
        self.active_prob = float(p.get("active_prob", 0.8))
        self.mode_probs = tuple(p.get("mode_probs", (0.4, 0.35, 0.25)))
        self.freq_range = tuple(p.get("freq_range", (0.2, 2.0)))
        self.step_period_range = tuple(p.get("step_period_range", (0.3, 0.8)))
        self.margin = float(p.get("limit_margin_rad", 0.05))

    def setup(self) -> None:
        env = self.env
        n, dev = env.num_envs, env.device
        arms = env._arm_idx
        lo = env.v2_pos_limits[arms, 0] + self.margin
        hi = env.v2_pos_limits[arms, 1] - self.margin
        default = env.default_dof_pos[0, arms]
        # symmetric usable half-range around the default pose, per arm joint
        self._lo, self._hi = lo, hi
        self._half = torch.minimum(hi - default, default - lo).clamp(min=0.0)
        self._default = default

        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.mode = torch.zeros(n, dtype=torch.long, device=dev)
        self.center = env.default_dof_pos[:, arms].clone()
        self.amp = torch.zeros(n, NUM_ARM_DOF, device=dev)
        self.freq = torch.zeros(n, device=dev)
        self.phase = torch.zeros(n, device=dev)
        self.step_period = torch.ones(n, device=dev)
        self.arm_target = env.default_dof_pos[:, arms].clone()
        self._t = torch.zeros(n, device=dev)
        self._next_step_t = torch.zeros(n, device=dev)
        self.reset(None)

    def reset(self, env_ids) -> None:
        env = self.env
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        k, dev = env_ids.numel(), env.device

        self.active[env_ids] = torch.rand(k, device=dev) < self.active_prob
        probs = torch.tensor(self.mode_probs, device=dev, dtype=torch.float)
        cutoffs = torch.cumsum(probs / probs.sum(), dim=0)
        self.mode[env_ids] = torch.searchsorted(
            cutoffs.contiguous(), torch.rand(k, device=dev).contiguous()
        ).clamp(max=len(self.mode_probs) - 1)
        self.freq[env_ids] = torch_rand_float(
            self.freq_range[0], self.freq_range[1], (k, 1), device=dev
        ).squeeze(1)
        self.phase[env_ids] = torch_rand_float(
            -math.pi, math.pi, (k, 1), device=dev
        ).squeeze(1)
        self.step_period[env_ids] = torch_rand_float(
            self.step_period_range[0], self.step_period_range[1], (k, 1), device=dev
        ).squeeze(1)
        self._t[env_ids] = 0.0
        self._next_step_t[env_ids] = self.step_period[env_ids]
        self._sample_pose(env_ids)
        self._write_target(env_ids)

    def _sample_pose(self, env_ids) -> None:
        dev = self.env.device
        k = env_ids.numel()
        span = 0.5 * float(self.env.arm_amplitude) * self._half.unsqueeze(0)
        self.center[env_ids] = self._default.unsqueeze(0) + (
            torch.rand(k, NUM_ARM_DOF, device=dev) * 2 - 1
        ) * span
        self.amp[env_ids] = torch.rand(k, NUM_ARM_DOF, device=dev) * span

    def _write_target(self, env_ids) -> None:
        mode = self.mode[env_ids]
        target = self.center[env_ids].clone()
        osc = torch.sin(
            2 * math.pi * self.freq[env_ids] * self._t[env_ids] + self.phase[env_ids]
        ).unsqueeze(1)
        target = torch.where(
            (mode == self.MODE_SIN).unsqueeze(1),
            self.center[env_ids] + self.amp[env_ids] * osc,
            target,
        )
        self.arm_target[env_ids] = torch.clamp(
            target, self._lo.unsqueeze(0), self._hi.unsqueeze(0)
        )

    def step(self) -> None:
        env = self.env
        self._t += env.dt
        due = (self.mode == self.MODE_STEP) & (self._t >= self._next_step_t)
        if bool(due.any()):
            idx = due.nonzero(as_tuple=False).flatten()
            self._sample_pose(idx)
            self._next_step_t[idx] = self._t[idx] + self.step_period[idx]
        self._write_target(torch.arange(env.num_envs, device=env.device, dtype=torch.long))


class ArmAmplitudeCurriculum(CurriculumTermBase):
    """Widen the sampled arm range 25% -> 100% as the policy survives longer.

    Same mechanism as `PenaltyCurriculum` (average episode length), because the
    same signal is what tells us the disturbance is being absorbed. If amplitude
    never widens, standing still became the cheaper policy — watch the logged
    `arm_amplitude` against `average_episode_length` (`docs/final_rl_policy.md` §8).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params or {}
        self.min_scale = float(p.get("min_scale", 0.25))
        self.max_scale = float(p.get("max_scale", 1.0))
        self.level_down_threshold = float(p.get("level_down_threshold", 300.0))
        self.level_up_threshold = float(p.get("level_up_threshold", 750.0))
        self.degree = float(p.get("degree", 0.001))
        env.arm_amplitude = max(float(p.get("initial_scale", 0.25)), self.min_scale)

    def setup(self) -> None:
        return

    def reset(self, env_ids) -> None:  # noqa: ARG002
        avg = float(self.env.average_episode_length)
        scale = float(self.env.arm_amplitude)
        if avg < self.level_down_threshold:
            scale *= 1.0 - self.degree
        elif avg > self.level_up_threshold:
            scale *= 1.0 + self.degree
        self.env.arm_amplitude = min(self.max_scale, max(self.min_scale, scale))

    def step(self) -> None:
        return


def upper_body_target(env) -> torch.Tensor:
    """Arm joint-position target relative to the default pose [num_envs, 14].

    Observations are computed *after* `command_manager.step()`, so for envs a
    skill owns this is the target `_pre_physics_step` is about to apply next —
    the policy can **anticipate** the motion instead of only reacting to the
    resulting `dof_pos` one step later. For envs where the policy owns its own
    arms it is the policy's own last applied target, which is information it
    already has (via `actions`) — so no channel is ever privileged or undefined.
    """
    arms = env._arm_idx
    own = env.action_manager.action[:, arms] * env.action_scales[arms].unsqueeze(0)
    upper = env.command_manager.get_state("upper_body_command")
    if upper is None:
        return own
    commanded = upper.arm_target - env.default_dof_pos[:, arms]
    return torch.where(upper.active.unsqueeze(1), commanded, own)


# ===========================================================================
# D · centroidal angular momentum
# ===========================================================================
def _body_masses(env) -> torch.Tensor:
    """[num_envs, num_bodies] link masses in the env's body order."""
    sim = env.simulator
    ids = torch.as_tensor(sim.body_ids, device=env.device, dtype=torch.long)
    robot = getattr(sim, "_robot", None)
    if robot is not None:  # IsaacSim / IsaacLab
        masses = robot.root_physx_view.get_masses().to(env.device).float()
        return masses[:, ids]
    backend = getattr(sim, "backend", None)
    if backend is not None:  # MuJoCo classic / MJWarp
        bridge = getattr(backend, "warp_model_bridge", None)
        if bridge is not None and hasattr(bridge, "body_mass"):
            return torch.as_tensor(bridge.body_mass, device=env.device).float()[:, ids]
        model_mass = torch.as_tensor(
            backend.model.body_mass, device=env.device, dtype=torch.float
        )
        return model_mass[ids].unsqueeze(0).expand(env.num_envs, -1).contiguous()
    raise RuntimeError(
        "CAM rewards need per-link masses; this simulator backend exposes none. "
        "Run the CAM stages on isaacsim (the default) or extend _body_masses()."
    )


def _compute_cam(env) -> torch.Tensor:
    """Mass-normalised centroidal angular momentum, world frame [num_envs, 3].

    Orbital term only — ``sum_i m_i (r_i - r_com) x (v_i - v_com)`` — divided by
    total mass so the value is invariant to the mass-randomisation curriculum.
    The link-spin term ``I_i w_i`` is deliberately omitted: it needs per-link
    inertia rotated into world frame every step (34 bodies x 3x3 per env) for a
    contribution the arm-swing signal does not depend on. Documented, not hidden.
    """
    if env._body_mass is None:
        env._body_mass = _body_masses(env)
    m = env._body_mass                                  # [N, B]
    pos = env.simulator._rigid_body_pos                 # [N, B, 3]
    vel = env.simulator._rigid_body_vel                 # [N, B, 3]
    total = m.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    com = (m.unsqueeze(-1) * pos).sum(dim=1) / total
    com_vel = (m.unsqueeze(-1) * vel).sum(dim=1) / total
    r = pos - com.unsqueeze(1)
    v = vel - com_vel.unsqueeze(1)
    return torch.cross(r, v, dim=-1).mul(m.unsqueeze(-1)).sum(dim=1) / total


def cam_tracking(env, sigma: float = 0.05, amplitude: float = CAM_REF_AMPLITUDE) -> torch.Tensor:
    """Track a gait-clock-derived vertical CAM reference.

    ``exp(-((k_ref - k_z) / (1 + |k_ref|))^2 / sigma)``. The reference oscillates
    with the left-leg gait phase and scales with commanded forward speed, so the
    optimum is an anti-phase arm swing against leg motion — and standing still
    asks for zero vertical CAM.

    ``sigma`` and `CAM_REF_AMPLITUDE` are **ASSUMED**: no reference motion exists
    for the A3, so there is nothing to derive them from. Check the logged `cam_z`
    on the first S2 run before trusting the gate — too small a sigma makes the
    term identically zero (no gradient at all), too large makes it uninformative.
    """
    gait = env.command_manager.get_state("locomotion_gait")
    phase = gait.phase[:, 0]
    speed = torch.norm(env.command_manager.commands[:, :2], dim=1)
    ref = amplitude * speed * torch.sin(phase)
    err = (ref - env.cam[:, 2]) / (1.0 + ref.abs())
    return torch.exp(-err.square() / sigma)


def penalty_cam_damping(
    env, cam_ref: float = 0.5, max_value: float = 5.0
) -> torch.Tensor:
    """Penalise *growth* of horizontal centroidal angular momentum.

    ``clamp(sum_{i in x,y} k_i * k_i_dot * dt / cam_ref^2, 0, max_value)`` — half
    the per-step increase in normalised squared horizontal CAM, which is positive
    exactly when |k_xy| is growing. That is what a perturbation injects.

    Two deliberate departures from `docs/final_rl_policy.md` §3D, which writes the
    term as ``-min(0, sum k_i k_i_dot)``:

    1. **Sign.** Reward terms carry their sign in the *weight* here, and
       ``-min(0, x) = relu(-x)`` is positive when momentum is *decaying* — with a
       negative weight that penalises exactly the behaviour we want. The term is
       non-negative and the weight is negative.
    2. **Scale.** ``k * k_dot`` is unbounded and has awkward units (m^4/s^3);
       measured raw values reach O(1e3) under a perturbation, which would swamp
       every other term. Normalising by ``cam_ref^2`` and multiplying by ``dt``
       makes it a bounded per-step fraction, clamped so a contact-solver spike
       cannot destroy an update.
    """
    growth = torch.sum(env.cam[:, :2] * env.cam_rate[:, :2], dim=1) * env.dt
    return (growth / (cam_ref * cam_ref)).clamp(0.0, max_value)


# ===========================================================================
# E · height scan
# ===========================================================================
def height_scan(
    env,
    noise_std: float = 0.02,
    dropout_prob: float = 0.05,
    outlier_prob: float = 0.02,
    outlier_scale: float = 0.3,
) -> torch.Tensor:
    """13 x 9 downward raycast grid in the base **yaw** frame [num_envs, 117].

    Row-major over (x forward, y lateral) so a `[N, 117]` view reshapes to the
    `(1, 13, 9)` `encoder_obs_shape` `CNNActor`/`CNNCritic` already expect.

    Noise, dropout and outliers are **not optional**: on hardware this comes from
    the head LiDAR's elevation map, which is neither perfect nor complete, and a
    policy trained on a clean map learns to trust one that cannot be delivered.
    They are disabled in eval so the sim2sim gate measures the policy, not the map.
    """
    if env._scan_cache_step == env.common_step_counter and env._scan_cache is not None:
        return env._scan_cache

    terrain = env.terrain_manager.get_state("locomotion_terrain")
    root = env.simulator.robot_root_states
    # yaw-only, matching TerrainLocomotion._get_base_heights: a full-quat rotation
    # would tilt the scan grid with the torso and read the wrong ground patch
    pts = env._scan_points_local.unsqueeze(0).expand(env.num_envs, -1, -1)
    world = quat_apply_yaw(_base_quat(env).repeat(1, SCAN_DIM), pts, True)
    world = world + root[:, :3].unsqueeze(1)

    ground = terrain.query_terrain_heights(world[..., :2].reshape(-1, 2))
    ground = ground.reshape(env.num_envs, SCAN_DIM)
    ground = torch.where(torch.isfinite(ground), ground, torch.zeros_like(ground))

    scan = (root[:, 2:3] - SCAN_NOMINAL_HEIGHT - ground).clamp(-SCAN_CLIP, SCAN_CLIP)

    if not env.is_evaluating:
        scan = scan + torch.randn_like(scan) * noise_std
        if outlier_prob > 0:
            hit = torch.rand_like(scan) < outlier_prob
            scan = torch.where(
                hit, scan + torch.randn_like(scan) * outlier_scale, scan
            )
        if dropout_prob > 0:
            drop = torch.rand_like(scan) < dropout_prob
            scan = torch.where(drop, torch.zeros_like(scan), scan)
        scan = scan.clamp(-SCAN_CLIP, SCAN_CLIP)

    env._scan_cache = scan
    env._scan_cache_step = env.common_step_counter
    return scan


# ===========================================================================
# F · smoothness + §4 reward terms
# ===========================================================================
def penalty_action_jerk(env) -> torch.Tensor:
    """Second-order action difference: penalise curvature, not motion.

    ``penalty_action_rate`` already penalises |a_t - a_{t-1}|, which taxes fast
    but *steady* motion. This taxes only abrupt changes of rate.
    """
    a = env.action_manager.action
    a1 = env.action_manager.prev_action
    a2 = env._action_tm2
    return torch.sum(torch.square(a - 2.0 * a1 + a2), dim=1)


def penalty_dof_acc(env, max_ratio: float = 10.0) -> torch.Tensor:
    """Joint-acceleration penalty, normalised by each joint's achievable maximum.

    Raw ``sum(acc^2)`` spans four orders of magnitude across this robot's joints
    (12 rad/s hips vs 22.7 rad/s waist roll) and is unbounded — a smoke run
    measured raw episode sums of 3e11 from contact spikes alone. Dividing by
    ``vel_limit / dt`` (the fastest one control step can change a joint) makes
    every joint contribute on the same O(1) scale, and the clamp keeps a solver
    spike from dominating an update.
    """
    acc_limit = env.v2_dof_vel_limits / env.dt
    ratio = (env.dof_acc / acc_limit).abs().clamp(max=max_ratio)
    return torch.sum(torch.square(ratio), dim=1)


def penalty_torques(env) -> torch.Tensor:
    """Torque penalty normalised by each joint's effort limit (6 Nm wrists vs 320 Nm knees)."""
    tau = env.action_manager.get_term("joint_control").torques
    return torch.sum(torch.square(tau / env.v2_torque_limits), dim=1)


def reward_feet_air_time(env, target_air_time: float = 0.4) -> torch.Tensor:
    """Reward long swing phases; gated off when no motion is commanded.

    Holosoma ships only `feet_phase` (a foot-height tracker), which shapes swing
    *height* but not swing *duration* — this is what removes the shuffling gait.
    """
    moving = torch.norm(env.command_manager.commands[:, :2], dim=1) > 0.1
    rew = torch.sum(
        (env._air_time_at_landing - target_air_time) * env.feet_first_contact.float(), dim=1
    )
    return rew * moving.float()


def penalty_feet_impact(
    env, soft_force_n: float = 900.0, max_excess: float = 5.0
) -> torch.Tensor:
    """Penalise the vertical touchdown force above a soft threshold.

    Impact penalties are what remove hard heel-strike; `feet_phase` alone will
    happily slam the foot down on the right trajectory.

    The threshold is ~1.5x body weight (60.18 kg -> 590 N), i.e. above a normal
    walking touchdown and below a slam. ``max_excess`` bounds the term at 25 per
    foot: contact forces are spiky and unbounded, and an unclamped square lets a
    single solver blowup dominate the whole update (measured raw episode sums of
    ~1e5 on a smoke run before this clamp existed).
    """
    fz = env.simulator.contact_forces[:, env.feet_indices, 2].abs()
    excess = ((fz - soft_force_n) / soft_force_n).clamp(0.0, max_excess)
    return torch.sum(torch.square(excess) * env.feet_first_contact.float(), dim=1)


# ===========================================================================
# B · FastSAC with a concurrent velocity estimator (and optional LCP)
# ===========================================================================
def _make_vel_est_actor(base_cls):
    """Wrap `Actor` or `CNNActor` with a base-linear-velocity estimator head.

    The head reads the actor's own (already-encoded) features, and its **detached**
    3-dim output is appended to the trunk input. Ground truth stays privileged —
    it is only ever a regression target — so the exported policy is deployable,
    and because the head lives *inside* the actor it travels in the ONNX.
    """

    class VelEstActor(base_cls):
        def _setup_network_with_input_dim(self, input_dim: int) -> None:
            layers, prev = [], input_dim
            for h in VEL_EST_HIDDEN:
                layers += [nn.Linear(prev, h, device=self.device), nn.SiLU()]
                prev = h
            layers += [nn.Linear(prev, VEL_EST_DIM, device=self.device)]
            self.est_net = nn.Sequential(*layers)
            super()._setup_network_with_input_dim(input_dim + VEL_EST_DIM)

        def _base_features(self, obs: torch.Tensor) -> torch.Tensor:
            return base_cls.process_obs(self, obs)

        def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
            x = self._base_features(obs)
            return torch.cat([x, self.est_net(x).detach()], -1)

        def velocity_estimate(self, obs: torch.Tensor) -> torch.Tensor:
            return self.est_net(self._base_features(obs))

    VelEstActor.__name__ = f"VelEst{base_cls.__name__}"
    return VelEstActor


def _group_term_layout(obs_manager, group_name: str) -> list[dict[str, Any]]:
    """[{name, dim, scale, start, width}] for a concatenated group, in real order.

    Real order == ``sorted(term_names)``, each term occupying ``dim *
    history_length`` contiguous columns, oldest frame first. This is the single
    source of truth for both the estimator's target slice and the ONNX layout
    metadata the sim2sim harness consumes — so a new observation term can never
    silently desynchronise them (`docs/final_rl_policy.md` §5, trap 1).
    """
    group_cfg = obs_manager.cfg.groups[group_name]
    hist = group_cfg.history_length
    out, cursor = [], 0
    for name in sorted(group_cfg.terms.keys()):
        term_cfg = group_cfg.terms[name]
        dim = int(obs_manager._compute_term(group_name, name, term_cfg).shape[1])
        width = dim * hist
        out.append(
            {
                "name": name,
                "dim": dim,
                "scale": term_cfg.scale,
                "noise": term_cfg.noise,
                "start": cursor,
                "width": width,
            }
        )
        cursor += width
    return out


class A3UltraFastSACAgent(FastSACAgent):
    """FastSAC + concurrent state estimation, Lipschitz ablation, layout metadata.

    Reached via ``algo._target_`` — `train_agent` resolves it with `get_class`, so
    **no Holosoma fork is required** for any of this.

    The estimator regresses the *normalised* privileged ``base_lin_vel`` slice of
    the critic observation. Regressing in normalised space keeps the head's target
    and the trunk's extra inputs O(1) without a hand-set scale; the logged error is
    converted back to m/s through the critic normaliser's own std, so the S1 gate
    ("estimator error logged and converging") reads in physical units.
    """

    ENABLE_VEL_ESTIMATOR = False
    ENABLE_LCP = False
    LCP_LAMBDA = LCP_LAMBDA

    def setup(self) -> None:
        super().setup()
        self._est_mse = torch.zeros((), device=self.device)
        self._lcp_value = torch.zeros((), device=self.device)
        self._vel_slice: slice | None = None
        self._vel_std = torch.ones(VEL_EST_DIM, device=self.device)

        if self.ENABLE_VEL_ESTIMATOR:
            self._locate_velocity_target()
            self.est_optimizer = optim.AdamW(
                list(self.actor.est_net.parameters()),
                lr=VEL_EST_LR,
                weight_decay=self.config.weight_decay,
                fused=True,
                betas=(0.9, 0.95),
            )
            logger.info(
                f"velocity estimator ON: critic slice {self._vel_slice} "
                f"(target term '{VEL_EST_TARGET_TERM}', normalised)"
            )
        if self.ENABLE_LCP:
            logger.info(f"Lipschitz policy penalty ON: lambda={self.LCP_LAMBDA}")

        self._patch_logging()

    # -- actor class selection (adds the estimator head) ----------------------
    def _actor_cls(self):
        base = CNNActor if self.config.use_cnn_encoder else Actor
        return _make_vel_est_actor(base) if self.ENABLE_VEL_ESTIMATOR else base

    def _locate_velocity_target(self) -> None:
        env = self.unwrapped_env
        keys = list(self.config.critic_obs_keys)
        if "critic_obs" not in keys:
            raise RuntimeError("velocity estimator expects a 'critic_obs' group")
        base = self.critic_obs_indices["critic_obs"]["start"]
        layout = _group_term_layout(env.observation_manager, "critic_obs")
        entry = next((t for t in layout if t["name"] == VEL_EST_TARGET_TERM), None)
        if entry is None:
            raise RuntimeError(
                f"velocity estimator needs the privileged '{VEL_EST_TARGET_TERM}' term "
                f"in critic_obs; found {[t['name'] for t in layout]}"
            )
        # newest frame is last within the term's history block
        end = base + entry["start"] + entry["width"]
        self._vel_slice = slice(end - VEL_EST_DIM, end)
        scale = entry["scale"]
        self._vel_obs_scale = float(scale if not isinstance(scale, (list, tuple)) else scale[0])

    # -- updates --------------------------------------------------------------
    def _update_pol(self, data):
        actor, args = self.actor, self.config
        actor_optimizer, scaler, qnet = self.actor_optimizer, self.scaler, self.qnet

        with self._maybe_amp():
            critic_observations = data["critic_observations"]
            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()
            q_outputs = qnet(critic_observations, actions)
            q_values = qnet.get_value(F.softmax(q_outputs, dim=-1))
            actor_loss = (self.log_alpha.exp().detach() * log_probs - q_values.mean(dim=0)).mean()

        if self.ENABLE_LCP:
            # Gradient penalty lambda * ||d a / d obs||^2. Computed in fp32 outside
            # autocast and added unscaled: GradScaler + create_graph is not a
            # supported combination, and this stage runs with compile=False.
            obs = data["observations"].detach().requires_grad_(True)
            _, mean, _ = actor(obs)
            grad = torch.autograd.grad(mean.sum(), obs, create_graph=True)[0]
            lcp = self.LCP_LAMBDA * grad.square().sum(dim=-1).mean()
            actor_loss = actor_loss + lcp
            self._lcp_value.copy_(lcp.detach().float())

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)
        scaler.unscale_(actor_optimizer)
        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_norm=args.max_grad_norm
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()

        if self.ENABLE_VEL_ESTIMATOR:
            self._update_estimator(data)

        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _update_estimator(self, data) -> None:
        # est_net receives no gradient from actor_loss (its output is detached into
        # the trunk), and actor_optimizer.zero_grad(set_to_none=True) above cleared
        # whatever this step leaves behind — so the two optimizers never collide.
        with self._maybe_amp():
            target = data["critic_observations"][:, self._vel_slice].detach()
            pred = self.actor.velocity_estimate(data["observations"])
            est_loss = F.mse_loss(pred, target)

        self.est_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(est_loss).backward()
        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.actor.est_net)
        self.scaler.unscale_(self.est_optimizer)
        self.scaler.step(self.est_optimizer)
        self.scaler.update()
        self._est_mse.copy_(est_loss.detach().float())

    # -- logging --------------------------------------------------------------
    def _patch_logging(self) -> None:
        original = self.logging_helper.post_epoch_logging

        def _with_extras(it, loss_dict, extra_log_dicts, **kwargs):
            merged = dict(loss_dict)
            merged.update(self._extra_metrics())
            return original(it=it, loss_dict=merged, extra_log_dicts=extra_log_dicts, **kwargs)

        self.logging_helper.post_epoch_logging = _with_extras

    def _extra_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.ENABLE_VEL_ESTIMATOR:
            std = torch.ones(1, device=self.device)
            norm = self.critic_obs_normalizer
            if hasattr(norm, "_std"):
                std = norm._std[0, self._vel_slice] + norm.eps
            # MSE in normalised units -> RMS error in m/s (undo obs scale too)
            rms = float(self._est_mse.sqrt().item()) * float(std.mean().item())
            out["vel_est_rms_ms"] = rms / max(self._vel_obs_scale, 1e-6)
            out["vel_est_mse_norm"] = float(self._est_mse.item())
        if self.ENABLE_LCP:
            out["lcp_penalty"] = float(self._lcp_value.item())
        return out

    # -- export: carry the observation layout with the policy -----------------
    def export(self, onnx_file_path: str) -> None:
        super().export(onnx_file_path)
        try:
            from holosoma.utils.inference_helpers import attach_onnx_metadata

            attach_onnx_metadata(
                onnx_path=onnx_file_path, metadata={"actor_obs_layout": self._obs_layout()}
            )
            self.logging_helper.save_to_wandb(onnx_file_path)
        except Exception as exc:  # pragma: no cover - never fail a run over metadata
            logger.warning(f"could not attach actor_obs_layout metadata: {exc}")

    def _obs_layout(self) -> dict[str, Any]:
        env = self.unwrapped_env
        sim_cfg = env.simulator.simulator_config.sim
        gait = env.command_manager.get_state("locomotion_gait")
        groups = []
        for key in self.config.actor_obs_keys:
            groups.append(
                {
                    "name": key,
                    "history_length": env.observation_manager.cfg.groups[key].history_length,
                    "terms": _group_term_layout(env.observation_manager, key),
                }
            )
        return {
            "groups": groups,
            "total_dim": self.actor_obs_dim,
            "control_dt": sim_cfg.control_decimation_steps / sim_cfg.fps,
            "clip_observations": env.observation_manager.cfg.clip_observations,
            "gait_period": getattr(gait, "gait_period", 1.0),
            "command_dim": int(env.command_manager.commands.shape[1]),
            "scandots": {
                "nx": SCAN_NX,
                "ny": SCAN_NY,
                "spacing": SCAN_SPACING,
                "x_offset": SCAN_X_OFFSET,
                "nominal_height": SCAN_NOMINAL_HEIGHT,
                "clip": SCAN_CLIP,
            },
            "arm_dof_indices": ARM_DOF_IDX,
            "extension": "a3_ultra_loco_v2",
        }


class A3UltraFastSACAgentVelEst(A3UltraFastSACAgent):
    ENABLE_VEL_ESTIMATOR = True


class A3UltraFastSACAgentLCP(A3UltraFastSACAgent):
    ENABLE_VEL_ESTIMATOR = True
    ENABLE_LCP = True


# `FastSACAgent.setup` hardcodes `(CNNActor, CNNCritic) if use_cnn_encoder else
# (Actor, Critic)`. Route the actor half through `_actor_cls()` so the estimator
# variant composes with the CNN encoder (S3+) instead of excluding it.
_ORIGINAL_SETUP = FastSACAgent.setup


def _setup_with_actor_hook(self):
    import holosoma.agents.fast_sac.fast_sac_agent as _mod

    if not hasattr(self, "_actor_cls"):
        return _ORIGINAL_SETUP(self)
    real_actor, real_cnn_actor = _mod.Actor, _mod.CNNActor
    chosen = self._actor_cls()
    _mod.Actor = _mod.CNNActor = chosen
    try:
        return _ORIGINAL_SETUP(self)
    finally:
        _mod.Actor, _mod.CNNActor = real_actor, real_cnn_actor


FastSACAgent.setup = _setup_with_actor_hook


# ===========================================================================
# Config builders
# ===========================================================================
def _actor_terms(*, heading: bool, arm_target: bool) -> dict[str, ObsTermCfg]:
    terms = {
        "base_ang_vel": ObsTermCfg(func=f"{_L}:base_ang_vel", scale=0.25, noise=0.0),
        "projected_gravity": ObsTermCfg(func=f"{_L}:projected_gravity", scale=1.0, noise=0.0),
        "command_lin_vel": ObsTermCfg(func=f"{_L}:command_lin_vel", scale=1.0, noise=0.0),
        "command_ang_vel": ObsTermCfg(func=f"{_L}:command_ang_vel", scale=1.0, noise=0.0),
        "dof_pos": ObsTermCfg(func=f"{_L}:dof_pos", scale=1.0, noise=0.01),
        "dof_vel": ObsTermCfg(func=f"{_L}:dof_vel", scale=0.05, noise=0.1),
        "actions": ObsTermCfg(func=f"{_L}:actions", scale=1.0, noise=0.0),
        "sin_phase": ObsTermCfg(func=f"{_L}:sin_phase", scale=1.0, noise=0.0),
        "cos_phase": ObsTermCfg(func=f"{_L}:cos_phase", scale=1.0, noise=0.0),
    }
    if heading:
        terms["heading_error"] = ObsTermCfg(func=f"{_X}:heading_error", scale=1.0, noise=0.02)
    if arm_target:
        terms["upper_body_target"] = ObsTermCfg(
            func=f"{_X}:upper_body_target", scale=1.0, noise=0.0
        )
    return terms


def build_observation(
    *, history_length: int = 1, heading: bool = False, arm_target: bool = False,
    scandots: bool = False,
) -> ObservationManagerCfg:
    actor = _actor_terms(heading=heading, arm_target=arm_target)
    critic = {k: replace(v, noise=0.0) for k, v in actor.items()}
    # privileged: the estimator's regression target lives here and NEVER in actor_obs
    critic["base_lin_vel"] = ObsTermCfg(func=f"{_L}:base_lin_vel", scale=2.0, noise=0.0)

    groups = {
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=history_length, terms=actor
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=history_length, terms=critic
        ),
    }
    if scandots:
        # history_length MUST stay 1: CNNActor reshapes this group to (1, 13, 9)
        groups["perception_obs"] = ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={"height_scan": ObsTermCfg(func=f"{_X}:height_scan", scale=1.0, noise=0.02)},
        )
    return ObservationManagerCfg(groups=groups)


def _pose_weights(*, arms_free: bool) -> list[float]:
    """v1's per-DOF pose weights, with the 14 arm entries optionally released.

    Releasing them is not optional once **C** lands: at 50.0 the policy is punished
    for arm deviation it no longer controls, and would fight the curriculum (and
    the CAM reward, which pushes the arms the other way).
    """
    weights = list(g1_29dof_loco_fast_sac.terms["pose"].params["pose_weights"])
    if arms_free:
        for i in ARM_DOF_IDX:
            weights[i] = 0.0
    return weights


def build_reward(
    *, arms_free: bool = False, cam: bool = False, smoothness: bool = False,
    gait_quality: bool = False,
) -> RewardManagerCfg:
    terms = dict(g1_29dof_loco_fast_sac.terms)
    terms["pose"] = replace(
        terms["pose"], params={"pose_weights": _pose_weights(arms_free=arms_free)}
    )
    if cam:
        terms["cam_tracking"] = RewardTermCfg(
            func=f"{_X}:cam_tracking", weight=1.0,
            params={"sigma": 0.05, "amplitude": CAM_REF_AMPLITUDE},
        )
        terms["penalty_cam_damping"] = RewardTermCfg(
            func=f"{_X}:penalty_cam_damping", weight=-0.5,
            params={"cam_ref": 0.5, "max_value": 5.0},
            tags=["penalty_curriculum"],
        )
    if gait_quality:
        terms["feet_air_time"] = RewardTermCfg(
            func=f"{_X}:reward_feet_air_time", weight=1.0, params={"target_air_time": 0.4}
        )
        terms["penalty_feet_impact"] = RewardTermCfg(
            func=f"{_X}:penalty_feet_impact", weight=-0.5,
            params={"soft_force_n": 900.0, "max_excess": 5.0},
            tags=["penalty_curriculum"],
        )
    if smoothness:
        terms["penalty_action_jerk"] = RewardTermCfg(
            func=f"{_X}:penalty_action_jerk", weight=-0.5, params={},
            tags=["penalty_curriculum"],
        )
        terms["penalty_dof_acc"] = RewardTermCfg(
            func=f"{_X}:penalty_dof_acc", weight=-0.05, params={"max_ratio": 10.0},
            tags=["penalty_curriculum"],
        )
        terms["penalty_torques"] = RewardTermCfg(
            func=f"{_X}:penalty_torques", weight=-0.1, params={},
            tags=["penalty_curriculum"],
        )
    return RewardManagerCfg(only_positive_rewards=False, terms=terms)


def build_command(
    *, heading: bool = False, arm_curriculum: bool = False, lin_vel_x=(-1.0, 1.0)
) -> CommandManagerCfg:
    loco_func = (
        f"{_X}:HeadingLocomotionCommand"
        if heading
        else "holosoma.managers.command.terms.locomotion:LocomotionCommand"
    )
    loco_params = {
        "command_ranges": {
            "lin_vel_x": list(lin_vel_x),
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.0, 1.0],
            "heading": [-3.14, 3.14],
        },
        "stand_prob": 0.2,
    }
    if heading:
        # ~20% of episodes stay on pure yaw-rate commands so spin-in-place does
        # not regress (docs/final_rl_policy.md §3A)
        loco_params.update({"heading_prob": 0.8, "heading_gain": 0.5, "heading_clip": 1.0})

    gait = CommandTermCfg(
        func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        params={"gait_period": 1.0, "gait_period_randomization_width": 0.2},
    )
    setup = {
        "locomotion_gait": gait,
        "locomotion_command": CommandTermCfg(func=loco_func, params=loco_params),
    }
    lifecycle = {
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait"
        ),
        "locomotion_command": CommandTermCfg(func=loco_func),
    }
    reset_terms, step_terms = dict(lifecycle), dict(lifecycle)
    if arm_curriculum:
        upper = CommandTermCfg(
            func=f"{_X}:UpperBodyCommand",
            params={
                "active_prob": 0.8,
                "mode_probs": [0.4, 0.35, 0.25],
                "freq_range": [0.2, 2.0],
                "step_period_range": [0.3, 0.8],
            },
        )
        setup["upper_body_command"] = upper
        reset_terms["upper_body_command"] = CommandTermCfg(func=f"{_X}:UpperBodyCommand")
        step_terms["upper_body_command"] = CommandTermCfg(func=f"{_X}:UpperBodyCommand")
    return CommandManagerCfg(
        params={"locomotion_command_resampling_time": 10.0},
        setup_terms=setup,
        reset_terms=reset_terms,
        step_terms=step_terms,
    )


def build_curriculum(*, arm_curriculum: bool = False) -> CurriculumManagerCfg:
    setup = {
        "average_episode_tracker": CurriculumTermCfg(
            func=f"{_C}:AverageEpisodeLengthTracker", params={}
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func=f"{_C}:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.5,
                "min_scale": 0.5,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.001,
            },
        ),
    }
    reset_terms: dict[str, CurriculumTermCfg] = {}
    if arm_curriculum:
        arm = CurriculumTermCfg(
            func=f"{_X}:ArmAmplitudeCurriculum",
            params={
                "initial_scale": 0.25,
                "min_scale": 0.25,
                "max_scale": 1.0,
                "level_down_threshold": 300.0,
                "level_up_threshold": 750.0,
                "degree": 0.001,
            },
        )
        setup["arm_amplitude"] = arm
        reset_terms["arm_amplitude"] = CurriculumTermCfg(func=f"{_X}:ArmAmplitudeCurriculum")
    return CurriculumManagerCfg(
        params={"num_compute_average_epl": 1000},
        setup_terms=setup,
        reset_terms=reset_terms,
        step_terms={},
    )


# Slope terrain (§4). v1 trained on flat 0.2 / rough 0.6 / low_obstacles 0.2 and
# smooth_slope == rough_slope == 0.0 — the policy has literally never seen a
# sustained grade, which is exactly where the alpine scenarios fail. Proportions
# are normalised by the terrain generator; these already sum to 1.0.
_slope_term = replace(
    terrain.terrain_locomotion_mix.terrain_term,
    terrain_config={
        "flat": 0.15,
        "rough": 0.35,
        "low_obstacles": 0.15,
        "smooth_slope": 0.20,
        "rough_slope": 0.15,
    },
    # spawning on a grade needs the real terrain height under the footprint, or
    # the robot starts interpenetrated and the episode dies before it acts
    spawn=replace(
        terrain.terrain_locomotion_mix.terrain_term.spawn,
        query_terrain_height=True,
        use_grid_sampling=True,
    ),
)
terrain_locomotion_slopes = replace(
    terrain.terrain_locomotion_mix, terrain_term=_slope_term
)


def build_algo(
    *, iterations: int, vel_estimator: bool = False, scandots: bool = False,
    lcp: bool = False, symmetry: bool = True, compile_: bool = True,
):
    if lcp:
        target = f"{_X}.A3UltraFastSACAgentLCP"
    elif vel_estimator:
        target = f"{_X}.A3UltraFastSACAgentVelEst"
    else:
        target = f"{_X}.A3UltraFastSACAgent"
    actor_keys = ["actor_obs"] + (["perception_obs"] if scandots else [])
    critic_keys = ["critic_obs"] + (["perception_obs"] if scandots else [])
    return replace(
        algo.fast_sac,
        _target_=target,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=iterations,
            use_symmetry=symmetry,
            use_cnn_encoder=scandots,
            encoder_obs_key="perception_obs",
            encoder_obs_shape=(1, SCAN_NX, SCAN_NY),
            actor_obs_keys=actor_keys,
            critic_obs_keys=critic_keys,
            save_interval=max(1000, iterations // 10),
            compile=compile_,
            # LCP needs a double backward; neither torch.compile nor GradScaler
            # handles create_graph, so that arm runs eager + fp32.
            amp=algo.fast_sac.config.amp and not lcp,
        ),
    )


def build_experiment(
    name: str,
    *,
    iterations: int,
    history_length: int = 1,
    heading: bool = False,
    vel_estimator: bool = False,
    arm_curriculum: bool = False,
    cam: bool = False,
    scandots: bool = False,
    slopes: bool = False,
    smoothness: bool = False,
    gait_quality: bool = False,
    lcp: bool = False,
    lin_vel_x=(-1.0, 1.0),
    compile_: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        env_class=f"{_X}.A3UltraLocoV2Manager",
        training=TrainingConfig(project="everest-a3", name=name),
        algo=build_algo(
            iterations=iterations,
            vel_estimator=vel_estimator,
            scandots=scandots,
            lcp=lcp,
            compile_=compile_,
        ),
        simulator=simulator.isaacsim,
        robot=a3_ultra_presets.a3_ultra_29dof,
        terrain=terrain_locomotion_slopes if slopes else terrain.terrain_locomotion_mix,
        observation=build_observation(
            history_length=history_length,
            heading=heading,
            arm_target=arm_curriculum,
            scandots=scandots,
        ),
        action=action.g1_29dof_joint_pos,
        termination=g1_29dof_termination,
        randomization=g1_29dof_randomization,
        command=build_command(
            heading=heading, arm_curriculum=arm_curriculum, lin_vel_x=lin_vel_x
        ),
        curriculum=build_curriculum(arm_curriculum=arm_curriculum),
        reward=build_reward(
            arms_free=arm_curriculum,
            cam=cam,
            smoothness=smoothness,
            gait_quality=gait_quality,
        ),
        nightly=NightlyConfig(
            iterations=iterations,
            metrics={
                "Episode/rew_tracking_lin_vel": [V1_TRACK_LIN, "inf"],
                "Episode/rew_tracking_ang_vel": [V1_TRACK_ANG, "inf"],
            },
        ),
    )


# ===========================================================================
# Stages (docs/final_rl_policy.md §6). Cost model: 0.195 s/iteration @ 4096 envs.
# ===========================================================================

# S0 — every feature OFF. The ONLY job is to prove the refactor is neutral before
# six behavioural changes land on top: 68/68 grid, lin-vel err <= 0.16. Not optional.
a3_ultra_loco_v2_s0 = EXPERIMENT_REGISTRY.add(
    "a3_ultra_loco_v2_s0",
    build_experiment("a3_ultra_loco_v2_s0", iterations=10_000),
)

# S1 — A heading + B estimator + G history(5), and the widened command envelope.
# Gate: yaw drift < 1 deg/s; speed within 5% at 0.5 and 1.0; rew_tracking_ang_vel
# >= 0.8; vel_est_rms_ms logged and falling.
a3_ultra_loco_v2_s1 = EXPERIMENT_REGISTRY.add(
    "a3_ultra_loco_v2_s1",
    build_experiment(
        "a3_ultra_loco_v2_s1",
        iterations=50_000,
        history_length=5,
        heading=True,
        vel_estimator=True,
        lin_vel_x=(-1.0, 1.5),
    ),
)

# S2 — C arm curriculum + D CAM (+ arm pose weights -> 0, same change).
# Gate: arm suite >= 26/28 WITHOUT observation masking; no regression on the grid.
a3_ultra_loco_v2_s2 = EXPERIMENT_REGISTRY.add(
    "a3_ultra_loco_v2_s2",
    build_experiment(
        "a3_ultra_loco_v2_s2",
        iterations=80_000,
        history_length=5,
        heading=True,
        vel_estimator=True,
        arm_curriculum=True,
        cam=True,
        lin_vel_x=(-1.0, 1.5),
    ),
)

# S3 — E height scan + slope terrain.
# Gate: clears the 3 alpine combos v1 fails; rough d1.0 lin-vel err <= 0.15.
# BLOCKING: the MuJoCo gate must raycast its own 13x9 scan first (built in S2).
a3_ultra_loco_v2_s3 = EXPERIMENT_REGISTRY.add(
    "a3_ultra_loco_v2_s3",
    build_experiment(
        "a3_ultra_loco_v2_s3",
        iterations=100_000,
        history_length=5,
        heading=True,
        vel_estimator=True,
        arm_curriculum=True,
        cam=True,
        scandots=True,
        slopes=True,
        gait_quality=True,
        lin_vel_x=(-1.0, 1.5),
    ),
)

# S4 — F smoothness, two arms of one ablation. Gate: jitter down >= 30% with
# tracking within 5% of S3. If S4 costs tracking, keep S3.
_s4_common = dict(
    iterations=30_000,
    history_length=5,
    heading=True,
    vel_estimator=True,
    arm_curriculum=True,
    cam=True,
    scandots=True,
    slopes=True,
    gait_quality=True,
    lin_vel_x=(-1.0, 1.5),
)
a3_ultra_loco_v2_s4 = EXPERIMENT_REGISTRY.add(
    "a3_ultra_loco_v2_s4",
    build_experiment("a3_ultra_loco_v2_s4", smoothness=True, **_s4_common),
)
# LCP arm: reported to cut action jitter 42.2 -> 3.2 against 5.7 for smoothness
# rewards -- but validated on PPO only, so it is an ablation, never load-bearing.
# compile=False: torch.compile does not handle the double backward this needs.
a3_ultra_loco_v2_s4_lcp = EXPERIMENT_REGISTRY.add(
    "a3_ultra_loco_v2_s4_lcp",
    build_experiment(
        "a3_ultra_loco_v2_s4_lcp", smoothness=False, lcp=True, compile_=False, **_s4_common
    ),
)

logger.info(
    "a3_ultra_loco_v2 registered: "
    + ", ".join(
        [
            "a3-ultra-loco-v2-s0",
            "a3-ultra-loco-v2-s1",
            "a3-ultra-loco-v2-s2",
            "a3-ultra-loco-v2-s3",
            "a3-ultra-loco-v2-s4",
            "a3-ultra-loco-v2-s4-lcp",
        ]
    )
)
