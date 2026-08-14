"""Sim2sim evaluation of an exported Holosoma locomotion policy in MuJoCo classic.

The policy was trained on IsaacSim/PhysX (`checkpoints/cloud_*`); this module is
the independent-physics gate. It reproduces Holosoma's runtime contract exactly
so the only difference between training and here is the physics engine:

* **Asset** — the generated training MJCF (`assets/a3_ultra/holosoma/*.xml`),
  the same 29-DOF robot the URDF gave IsaacSim.
* **Rates** — 50 Hz policy (Holosoma: sim 200 Hz / control_decimation 4). MuJoCo
  integrates at the MJCF's 1 ms timestep with decimation 20 (the manifest's
  documented evaluation stack: finer substep, same policy rate).
* **Observation** — the actor group is concatenated in **alphabetically sorted
  term order** (`ObservationManager.compute_group`), not config order:

  | slice | term | dims | scale |
  | --- | --- | --- | --- |
  | 0:29 | actions (previous raw policy output) | 29 | 1.0 |
  | 29:32 | base_ang_vel (body frame) | 3 | 0.25 |
  | 32:33 | command_ang_vel (yaw rate) | 1 | 1.0 |
  | 33:35 | command_lin_vel (vx, vy) | 2 | 1.0 |
  | 35:37 | cos_phase (per-leg gait clock) | 2 | 1.0 |
  | 37:66 | dof_pos - default | 29 | 1.0 |
  | 66:95 | dof_vel | 29 | 0.05 |
  | 95:98 | projected_gravity | 3 | 1.0 |
  | 98:100 | sin_phase | 2 | 1.0 |

  The gait phase is a 2-vector (one entry per leg). In eval mode Holosoma pins
  the offsets to ``[0, -pi]`` and the frequency to ``1/gait_period``; the phase
  advances as ``k * 2*pi*dt*gait_freq + offset`` wrapped to ``[-pi, pi)``, and is
  forced to ``pi`` on both legs whenever the command is ~zero (stand).
* **Action** — the ONNX embeds the observation normalizer and the per-joint
  action bounds, so its output is the *raw* action; the environment applies
  ``target = default_pose + action_scale * action`` and a PD law with the
  metadata's kp/kd, torque-clipped to the effort limits.

Everything above is verified against holosoma `6e146b0` source, not docs:
`managers/observation/manager.py`, `managers/observation/terms/locomotion.py`,
`managers/command/terms/locomotion.py`, `managers/action/terms/joint_control.py`,
`agents/fast_sac/fast_sac_agent.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from everest_locomotion import REPO_ROOT
from everest_locomotion.robots.manifest import RobotManifest
from everest_locomotion.terrains.spec import TerrainPatch

HOLOSOMA_MJCF = REPO_ROOT / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.xml"

# Holosoma runtime constants for this experiment (holosoma_config.yaml of the run).
CONTROL_DT = 0.02          # sim.fps 200 / control_decimation 4
GAIT_PERIOD = 1.0          # command.setup_terms.locomotion_gait.gait_period
STAND_PHASE_VALUE = math.pi
CLIP_OBSERVATIONS = 100.0
ACTION_CLIP_VALUE = 100.0
SPAWN_HEIGHT = 1.07        # robot.init_state.pos[2]


# ---------------------------------------------------------------------------
# policy


class HolosomaPolicy:
    """Exported FastSAC actor + the observation assembly it was trained with."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.path = Path(onnx_path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.obs_dim = int(self.session.get_inputs()[0].shape[-1])

        meta = self.session.get_modelmeta().custom_metadata_map
        self.dof_names: list[str] = json.loads(meta["dof_names"])
        self.kp = np.asarray(json.loads(meta["kp"]), dtype=np.float64)
        self.kd = np.asarray(json.loads(meta["kd"]), dtype=np.float64)
        scale = json.loads(meta["action_scale"])
        self.action_scale = np.asarray(
            scale if isinstance(scale, list) else [scale] * len(self.dof_names), dtype=np.float64
        )
        self.iteration = int(meta.get("iteration", -1))
        self.command_ranges = json.loads(meta.get("command_ranges", "{}"))

        cfg = json.loads(meta["experiment_config"])
        self.default_dof_pos = np.array(
            [cfg["robot"]["init_state"]["default_joint_angles"][n] for n in self.dof_names],
            dtype=np.float64,
        )
        self.n_dof = len(self.dof_names)

        expected = 3 * self.n_dof + 3 + 1 + 2 + 2 + 3 + 2
        if self.obs_dim != expected:
            raise ValueError(
                f"{self.path.name}: actor obs dim {self.obs_dim} != expected {expected}; "
                "the observation term set changed — re-derive the layout before trusting results."
            )

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS).astype(np.float32)[None]
        return self.session.run(None, {self.input_name: x})[0][0].astype(np.float64)


# ---------------------------------------------------------------------------
# scene


def build_model(
    patch: TerrainPatch | None = None,
    mjcf_path: Path = HOLOSOMA_MJCF,
) -> mujoco.MjModel:
    """Load the training MJCF, optionally swapping its floor for a heightfield."""
    if patch is None:
        return mujoco.MjModel.from_xml_path(str(mjcf_path))

    import xml.etree.ElementTree as ET

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") == "floor":
            worldbody.remove(geom)

    h = patch.heights_m
    h_min, h_max = float(h.min()), float(h.max())
    elevation = max(h_max - h_min, 1e-6)
    nx, ny = h.shape
    ET.SubElement(
        root.find("asset"),
        "hfield",
        name="terrain",
        nrow=str(ny),
        ncol=str(nx),
        size=f"{patch.spec.size_m[0] / 2} {patch.spec.size_m[1] / 2} {elevation} 0.5",
    )
    friction = 1.0 if patch.friction is None else float(np.mean(patch.friction))
    ET.SubElement(
        worldbody,
        "geom",
        name="terrain",
        type="hfield",
        hfield="terrain",
        conaffinity="7",
        friction=f"{friction} 0.005 0.0001",
        pos=f"0 0 {h_min}",
    )

    # Written next to the source so relative mesh paths still resolve.
    out = mjcf_path.with_name(mjcf_path.stem + "_scene_hfield.xml")
    tree.write(out)
    try:
        model = mujoco.MjModel.from_xml_path(str(out))
    finally:
        out.unlink(missing_ok=True)

    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    assert hid >= 0
    model.hfield_data[:] = ((h - h_min) / elevation).T.ravel()
    return model


# ---------------------------------------------------------------------------
# rollout


@dataclass
class Push:
    time_s: float
    velocity_xyz: tuple[float, float, float]


@dataclass
class Command:
    """Velocity command, optionally switching partway through the episode."""

    lin_vel_x: float = 0.0
    lin_vel_y: float = 0.0
    ang_vel_yaw: float = 0.0
    schedule: list[tuple[float, tuple[float, float, float]]] = field(default_factory=list)

    def at(self, t: float) -> np.ndarray:
        cmd = (self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw)
        for start, value in self.schedule:
            if t >= start:
                cmd = value
        return np.asarray(cmd, dtype=np.float64)


@dataclass
class RolloutResult:
    name: str = ""
    survived: bool = True
    fall_time_s: float | None = None
    duration_s: float = 0.0
    steps: int = 0
    mean_base_height: float = 0.0
    min_base_height: float = 0.0
    max_tilt_deg: float = 0.0
    mean_tilt_deg: float = 0.0
    lin_vel_error: float = 0.0        # mean |v_body_xy - cmd_xy| over the tracked window
    ang_vel_error: float = 0.0
    mean_speed_ms: float = 0.0
    distance_travelled_m: float = 0.0
    heading_drift_deg: float = 0.0   # unwrapped yaw travelled minus commanded yaw
    slip_distance_m: float = 0.0
    torque_saturation_frac: float = 0.0
    mean_abs_torque_nm: float = 0.0
    action_jitter: float = 0.0        # mean |a_t - a_{t-1}| in raw action units
    joint_limit_frac: float = 0.0
    cost_of_transport: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = dict(self.__dict__)
        row.pop("extras", None)
        return row


class A3Sim:
    """One MuJoCo episode of the exported policy on a given scene."""

    def __init__(
        self,
        policy: HolosomaPolicy,
        manifest: RobotManifest,
        model: mujoco.MjModel | None = None,
        patch: TerrainPatch | None = None,
        friction: float | None = None,
        added_base_mass_kg: float = 0.0,
    ):
        self.policy = policy
        self.manifest = manifest
        self.patch = patch
        self.model = model if model is not None else build_model(patch)
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.decimation = int(round(CONTROL_DT / m.opt.timestep))
        if abs(self.decimation * m.opt.timestep - CONTROL_DT) > 1e-9:
            raise ValueError("MJCF timestep does not divide the 20 ms control period")

        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in policy.dof_names]
        if min(jids) < 0:
            missing = [n for n, i in zip(policy.dof_names, jids) if i < 0]
            raise ValueError(f"joints missing from MJCF: {missing}")
        self.qpos_adr = np.array([m.jnt_qposadr[i] for i in jids])
        self.qvel_adr = np.array([m.jnt_dofadr[i] for i in jids])
        self.act_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in policy.dof_names]
        )
        if (self.act_ids < 0).any():
            raise ValueError("MJCF is missing motor actuators for the policy joints")

        self.effort_limit = np.array(
            [manifest.joints[n]["effort_nm"] for n in policy.dof_names], dtype=np.float64
        )
        self.joint_limits = np.array(
            [manifest.joints[n]["limits_rad"] for n in policy.dof_names], dtype=np.float64
        )
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.feet_bodies = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in ("left_ankle_roll_Link", "right_ankle_roll_Link")
        ]

        if friction is not None:
            # MuJoCo combines pairwise friction with max(), so both sides must drop.
            self.model.geom_friction[:, 0] = friction
        if added_base_mass_kg:
            self.model.body_mass[self.base_body] += added_base_mass_kg

        self.phase_dt = 2.0 * math.pi * CONTROL_DT / GAIT_PERIOD

    # -- state helpers ------------------------------------------------------
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    def _rot(self) -> np.ndarray:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        return rot.reshape(3, 3)

    def _yaw(self) -> float:
        fwd = self._rot()[:2, 0]
        return math.atan2(float(fwd[1]), float(fwd[0]))

    def projected_gravity(self) -> np.ndarray:
        return self._rot().T @ np.array([0.0, 0.0, -1.0])

    def base_lin_vel_body(self) -> np.ndarray:
        return self._rot().T @ self.data.qvel[0:3]

    def base_ang_vel_body(self) -> np.ndarray:
        # MuJoCo free joints express rotational velocity in the body frame already.
        return self.data.qvel[3:6].copy()

    def terrain_height(self, x: float, y: float) -> float:
        return 0.0 if self.patch is None else self.patch.height_at(x, y)

    def phase(self, step: int, cmd: np.ndarray) -> np.ndarray:
        offset = np.array([0.0, -math.pi])
        ph = np.fmod(step * self.phase_dt + offset + math.pi, 2 * math.pi) - math.pi
        if step >= 1 and np.linalg.norm(cmd[:2]) < 0.01 and abs(cmd[2]) < 0.01:
            ph = np.full(2, STAND_PHASE_VALUE)
        return ph

    def _min_contact_dist(self) -> float:
        mujoco.mj_forward(self.model, self.data)
        if self.data.ncon == 0:
            return float("inf")
        return min(float(self.data.contact[c].dist) for c in range(self.data.ncon))

    def _clear_spawn_height(self, z_nominal: float, margin_m: float = 0.005) -> float:
        """Lowest spawn height at which no geom is already penetrating the ground.

        Spawning at the terrain height under the *pelvis* buries the toe of a
        0.27 m foot by ~2 cm on a 10° grade; MuJoCo resolves that penetration
        explosively and the episode dies before the policy ever acts. A fixed
        footprint-max clearance overcorrects instead (a 13 cm drop onto a 15°
        slope is its own destabiliser), so search for the actual contact height.
        """
        if self.patch is None:
            return z_nominal
        for z in np.arange(z_nominal - 0.10, z_nominal + 0.45, 0.005):
            self.data.qpos[2] = z
            if self._min_contact_dist() > -1e-4:
                return float(z) + margin_m
        return z_nominal + 0.45

    def reset(self, yaw: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = math.cos(yaw / 2)
        self.data.qpos[6] = math.sin(yaw / 2)
        self.data.qpos[self.qpos_adr] = self.policy.default_dof_pos
        self.data.qpos[2] = self._clear_spawn_height(
            SPAWN_HEIGHT + self.terrain_height(0.0, 0.0)
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # -- observation --------------------------------------------------------
    def observe(self, prev_action: np.ndarray, step: int, cmd: np.ndarray) -> np.ndarray:
        dof_pos = self.data.qpos[self.qpos_adr] - self.policy.default_dof_pos
        dof_vel = self.data.qvel[self.qvel_adr]
        ph = self.phase(step, cmd)
        return np.concatenate(
            [
                prev_action,                       # actions
                self.base_ang_vel_body() * 0.25,   # base_ang_vel
                cmd[2:3],                          # command_ang_vel
                cmd[0:2],                          # command_lin_vel
                np.cos(ph),                        # cos_phase
                dof_pos,                           # dof_pos
                dof_vel * 0.05,                    # dof_vel
                self.projected_gravity(),          # projected_gravity
                np.sin(ph),                        # sin_phase
            ]
        )

    # -- episode ------------------------------------------------------------
    def run(
        self,
        duration_s: float = 10.0,
        command: Command | None = None,
        pushes: list[Push] | None = None,
        track_after_s: float = 1.0,
        renderer=None,
        camera=None,
        video_fps: int = 30,
        frame_cb=None,
        name: str = "",
        obs_noise: float = 0.0,
        seed: int = 0,
        spawn_yaw: float = 0.0,
        fall_tilt_deg: float = 70.0,
        fall_height_m: float = 0.45,
    ) -> RolloutResult:
        rng = np.random.default_rng(seed)
        command = command or Command()
        pushes = sorted(pushes or [], key=lambda p: p.time_s)
        self.reset(yaw=spawn_yaw)

        n_control = int(round(duration_s / CONTROL_DT))
        render_every = 0
        if renderer is not None:
            render_every = max(1, int(round((1.0 / video_fps) / CONTROL_DT)))
            if abs(render_every * CONTROL_DT * video_fps - 1.0) > 1e-9:
                raise ValueError(
                    f"video_fps={video_fps} does not divide the {1 / CONTROL_DT:g} Hz "
                    "control rate — playback would not be real time"
                )

        prev_action = np.zeros(self.policy.n_dof)
        push_i = 0
        heights, tilts, torques, jitter = [], [], [], []
        sat, lim_frac, slip = [], [], 0.0
        vel_err, ang_err, speeds = [], [], []
        mech_power = 0.0
        fall_time = None
        start_xy = self.base_pos()[:2].copy()
        prev_foot_xy = None
        yaw_travelled = 0.0
        yaw_commanded = 0.0
        prev_yaw = self._yaw()

        for step in range(n_control):
            t = step * CONTROL_DT
            cmd = command.at(t)

            obs = self.observe(prev_action, step, cmd)
            if obs_noise > 0.0:
                obs = obs + rng.uniform(-obs_noise, obs_noise, obs.shape)
            action = self.policy.act(obs)
            jitter.append(float(np.abs(action - prev_action).mean()))
            prev_action = action

            target = self.policy.default_dof_pos + self.policy.action_scale * np.clip(
                action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE
            )

            while push_i < len(pushes) and t >= pushes[push_i].time_s:
                self.data.qvel[0:3] += np.asarray(pushes[push_i].velocity_xyz)
                push_i += 1

            for _ in range(self.decimation):
                q = self.data.qpos[self.qpos_adr]
                qd = self.data.qvel[self.qvel_adr]
                tau = np.clip(
                    self.policy.kp * (target - q) - self.policy.kd * qd,
                    -self.effort_limit,
                    self.effort_limit,
                )
                self.data.ctrl[self.act_ids] = tau
                mujoco.mj_step(self.model, self.data)
                mech_power += float(np.abs(tau * qd).sum()) * self.model.opt.timestep

            if not np.isfinite(self.data.qpos).all():
                fall_time = t
                break

            grav = self.projected_gravity()
            tilt = math.degrees(math.acos(float(np.clip(-grav[2], -1.0, 1.0))))
            pos = self.base_pos()
            z = pos[2] - self.terrain_height(pos[0], pos[1])
            tilts.append(tilt)
            heights.append(z)

            torques.append(float(np.abs(tau).mean()))
            sat.append(float(np.mean(np.abs(tau) >= 0.97 * self.effort_limit)))
            q = self.data.qpos[self.qpos_adr]
            lim_frac.append(
                float(
                    np.mean(
                        (q <= self.joint_limits[:, 0] + 0.02) | (q >= self.joint_limits[:, 1] - 0.02)
                    )
                )
            )

            foot_xy = np.array([self.data.xpos[b][:2] for b in self.feet_bodies])
            if prev_foot_xy is not None:
                for i, b in enumerate(self.feet_bodies):
                    if self._body_in_contact(b):
                        slip += float(np.linalg.norm(foot_xy[i] - prev_foot_xy[i]))
            prev_foot_xy = foot_xy

            yaw_now = self._yaw()
            yaw_travelled += (yaw_now - prev_yaw + math.pi) % (2 * math.pi) - math.pi
            prev_yaw = yaw_now
            yaw_commanded += cmd[2] * CONTROL_DT

            v_body = self.base_lin_vel_body()
            w_body = self.base_ang_vel_body()
            speeds.append(float(np.linalg.norm(v_body[:2])))
            if t >= track_after_s:
                vel_err.append(float(np.linalg.norm(v_body[:2] - cmd[:2])))
                ang_err.append(abs(float(w_body[2] - cmd[2])))

            if tilt > fall_tilt_deg or z < fall_height_m:
                fall_time = t
                if renderer is not None and frame_cb is not None:
                    frame_cb(self._render(renderer, camera), t, cmd, v_body, w_body, tilt, z, False)
                break

            if renderer is not None and frame_cb is not None and step % render_every == 0:
                frame_cb(self._render(renderer, camera), t, cmd, v_body, w_body, tilt, z, True)

        pos = self.base_pos()
        steps_done = len(heights)
        mass = float(self.model.body_subtreemass[self.base_body])
        dist = float(np.linalg.norm(pos[:2] - start_xy))

        return RolloutResult(
            name=name,
            survived=fall_time is None,
            fall_time_s=fall_time,
            duration_s=duration_s if fall_time is None else fall_time,
            steps=steps_done,
            mean_base_height=float(np.mean(heights)) if heights else 0.0,
            min_base_height=float(np.min(heights)) if heights else 0.0,
            max_tilt_deg=float(np.max(tilts)) if tilts else 180.0,
            mean_tilt_deg=float(np.mean(tilts)) if tilts else 180.0,
            lin_vel_error=float(np.mean(vel_err)) if vel_err else float("nan"),
            ang_vel_error=float(np.mean(ang_err)) if ang_err else float("nan"),
            mean_speed_ms=float(np.mean(speeds)) if speeds else 0.0,
            distance_travelled_m=dist,
            heading_drift_deg=math.degrees(yaw_travelled - yaw_commanded),
            slip_distance_m=slip,
            torque_saturation_frac=float(np.mean(sat)) if sat else 0.0,
            mean_abs_torque_nm=float(np.mean(torques)) if torques else 0.0,
            action_jitter=float(np.mean(jitter)) if jitter else 0.0,
            joint_limit_frac=float(np.mean(lim_frac)) if lim_frac else 0.0,
            cost_of_transport=(
                mech_power / (mass * 9.81 * dist) if dist > 0.25 else float("nan")
            ),
        )

    def _body_in_contact(self, body_id: int) -> bool:
        m, d = self.model, self.data
        for c in range(d.ncon):
            con = d.contact[c]
            if m.geom_bodyid[con.geom1] == body_id or m.geom_bodyid[con.geom2] == body_id:
                return True
        return False

    def _render(self, renderer, camera):
        pos = self.base_pos()
        camera.lookat[:] = [pos[0], pos[1], pos[2] - 0.25]
        renderer.update_scene(self.data, camera)
        return renderer.render()
