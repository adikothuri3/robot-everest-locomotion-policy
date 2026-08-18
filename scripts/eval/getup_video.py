"""Roll out an exported get-up policy from the fallen-pose bank and record video.

The locomotion suite (`sim2sim_suite.py`) cannot grade get-up: every scenario it
owns spawns the robot standing and drives it with velocity commands, and the
get-up actor has neither command nor gait-phase observations. This script is the
get-up equivalent — same MuJoCo physics, same PD/action contract, but episodes
start from a real fallen pose in `assets/a3_ultra/getup/` and success is the
manifest's `getup.terminal` gate.

    python scripts/eval/getup_video.py --onnx checkpoints/v1_getup_28k_plateau/model_27999.onnx
    python scripts/eval/getup_video.py --onnx <ckpt> --postures supine --episodes 8
    python scripts/eval/getup_video.py --onnx <ckpt> --action-authority 1.0   # deploy scale

Success requires *holding* the handoff pose (the locomotion policy's expected
initial state) for `hold_time_s`, not merely passing through it — a robot that
lurches upright and topples is a failure, which is the distinction that matters
when reading a get-up video.

**The action map is not in the ONNX.** `A3UltraGetupManager._pre_physics_step`
multiplies every action by `getup_action_authority` (2.0 early, annealing to the
deployable 1.0) and the wrists by `getup.wrist_action_scale / action_scale`
before the action manager ever sees it. That factor is runtime curriculum state,
so it lives in the `.pt` and not in the config the ONNX embeds — replaying a
checkpoint at authority 1.0 when it was trained at 2.0 commands **half** the
joint displacement it learned to produce, which reads as "rises to a kneel and
stalls" no matter how good the policy is. This script therefore defaults to the
authority the checkpoint was trained at and prints where that number came from.
Pass `--action-authority 1.0` to ask the separate question of whether the policy
is deployable at the manifest's action scale.
"""

from __future__ import annotations

import argparse
import json
import math
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
    build_model,
)
from everest_locomotion.robots.manifest import load_manifest

POSE_DIR = REPO_ROOT / "assets" / "a3_ultra" / "getup"
VIDEO_W, VIDEO_H, VIDEO_FPS = 960, 540, 25
POSTURES = ["supine", "prone", "side_left", "side_right"]


# ---------------------------------------------------------------------------
# pose bank


def load_poses(posture: str) -> np.ndarray:
    path = POSE_DIR / f"fallen_poses_{posture}.npy"
    if not path.exists():
        raise SystemExit(
            f"no pose bank at {path} — generate it with "
            "scripts/getup/generate_fallen_poses.py"
        )
    return np.load(path)


class GetupSim(A3Sim):
    """`A3Sim` with the episode starting from a fallen pose instead of standing."""

    def reset_fallen(self, qpos: np.ndarray) -> None:
        """Drop the robot into a settled fallen pose.

        The bank stores a full `qpos` (base pos + quat + 29 hinges in MJCF *tree*
        order), captured after the pose settled under gravity, so it is written
        straight in rather than reassembled through the canonical joint order.
        """
        mujoco.mj_resetData(self.model, self.data)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(
                f"pose bank has nq={qpos.shape[0]} but the MJCF has {self.model.nq} "
                "— the bank was generated against a different asset; regenerate it"
            )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self._obs_history = {}
        mujoco.mj_forward(self.model, self.data)


# ---------------------------------------------------------------------------
# the action map the checkpoint was trained under


def training_curriculum_state(onnx_path: Path) -> dict:
    """Read the get-up curriculum state the run saved next to the ONNX.

    `A3UltraGetupManager.get_checkpoint_state` writes `getup_action_authority`
    and `getup_assist_scale` into the `.pt`; the ONNX export carries the
    experiment *config* but not this, because it is state the curriculum
    advanced at runtime. The `.pt` is the only in-repo record of the action map
    a given checkpoint was trained under, so read it when we can. Best effort:
    the eval venv has no torch, and grading must still work there.
    """
    pt = onnx_path.with_suffix(".pt")
    if not pt.exists():
        return {}
    try:
        import torch
    except ImportError:
        return {}
    wanted = ("getup_action_authority", "getup_assist_scale")
    found: dict[str, float] = {}

    def walk(node) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in wanted:
                try:
                    found[key] = float(value)
                except (TypeError, ValueError):
                    pass
            else:
                walk(value)

    walk(torch.load(pt, map_location="cpu", weights_only=False))
    return found


def resolve_action_map(policy: HolosomaPolicy, manifest, override: float | None) -> dict:
    """Decide the authority / wrist factor this rollout replays, and say why."""
    trained = training_curriculum_state(policy.path)
    if override is not None:
        authority, source = override, "--action-authority"
    elif "getup_action_authority" in trained:
        authority = trained["getup_action_authority"]
        source = f"{policy.path.with_suffix('.pt').name} (as trained)"
    elif policy.layout.has("action_authority"):
        # The policy observes beta, so the curriculum existed in the run that
        # produced it and its value is load-bearing. Guessing here is what makes
        # a working policy look broken -- refuse instead.
        raise SystemExit(
            f"{policy.path.name} observes `action_authority`, so it was trained under the\n"
            "authority curriculum, but the value is not recoverable: no readable\n"
            f"{policy.path.with_suffix('.pt').name} beside it (needs torch on the path).\n"
            "Pass --action-authority explicitly — read Env/getup_action_authority off the\n"
            "run's TensorBoard, or 1.0 to grade deployability at the manifest action scale."
        )
    else:
        # Pre-v3 checkpoint: no authority curriculum existed, actions went
        # through unscaled.
        authority, source = 1.0, "1.0 (checkpoint predates the authority curriculum)"

    # Wrist soft-freeze is config, not curriculum: derive it rather than
    # duplicating WRIST_ACTION_FACTOR from the training extension.
    wrist_idx = [i for i, n in enumerate(policy.dof_names) if "wrist" in n]
    wrist_factor = 1.0
    if wrist_idx:
        wrist_factor = float(
            manifest.raw["getup"]["wrist_action_scale"]
            / np.mean(policy.action_scale[wrist_idx])
        )
    return {
        "action_authority": float(authority),
        "authority_source": source,
        "wrist_action_factor": wrist_factor,
        "trained_assist_scale": trained.get("getup_assist_scale"),
    }


# ---------------------------------------------------------------------------
# success gate


class Terminal:
    """The manifest's get-up terminal gate, evaluated per control step."""

    def __init__(self, manifest, policy: HolosomaPolicy):
        cfg = manifest.raw["getup"]["terminal"]
        self.height = float(manifest.raw["default_pose"]["base_height_m"])
        self.height_tol = float(cfg["base_height_tol_m"])
        self.pose_tol = float(cfg["joint_pos_tol_rad"])
        self.lin_vel_max = float(cfg["base_lin_vel_max_m_s"])
        self.joint_vel_max = float(cfg["joint_vel_max_rad_s"])
        self.hold_time = float(cfg["hold_time_s"])
        # "mean over legs+waist" — the joints that define the stance. Arms are
        # excluded: the handoff contract does not care where the arms ended up.
        legs_waist = [
            i for i, n in enumerate(policy.dof_names)
            if any(k in n for k in ("hip", "knee", "ankle", "waist"))
        ]
        self.pose_idx = np.asarray(legs_waist, dtype=int)
        self.default = policy.default_dof_pos

    def check(self, sim: GetupSim) -> tuple[bool, dict]:
        q = sim.data.qpos[sim.qpos_adr]
        qd = sim.data.qvel[sim.qvel_adr]
        z = float(sim.base_pos()[2])
        pose_err = float(
            np.abs(q[self.pose_idx] - self.default[self.pose_idx]).mean()
        )
        lin = float(np.linalg.norm(sim.data.qvel[0:3]))
        # legs+waist, matching `_task_success_condition` in the training env and
        # the manifest's "mean over legs+waist" scope. Over all 29 joints an arm
        # still settling fails the gate on a stance the handoff contract accepts.
        jv = float(np.abs(qd[self.pose_idx]).max())
        ok = (
            abs(z - self.height) <= self.height_tol
            and pose_err <= self.pose_tol
            and lin <= self.lin_vel_max
            and jv <= self.joint_vel_max
        )
        return ok, {"z": z, "pose_err": pose_err, "lin": lin, "jvel": jv}


# ---------------------------------------------------------------------------
# rendering


def make_font(size: int):
    from PIL import ImageFont

    for name in ("segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Overlay:
    def __init__(self, title: str, caption: str, policy_label: str):
        self.title, self.caption, self.policy_label = title, caption, policy_label
        self.title_font, self.font, self.small = make_font(24), make_font(18), make_font(15)
        self.frames: list[np.ndarray] = []

    def __call__(self, frame, t, m: dict, held: float, done: bool):
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, img.width, 92], fill=(12, 14, 18, 190))
        d.text((16, 10), self.title, font=self.title_font, fill=(245, 245, 245))
        d.text((16, 42), self.caption, font=self.small, fill=(170, 200, 235))
        d.text((16, 64), f"A3 Ultra · get-up · {self.policy_label} · MuJoCo (sim2sim)",
               font=self.small, fill=(140, 150, 165))

        lines = [
            f"t        {t:5.2f} s",
            f"pelvis   {m['z']:.2f} m   (target {m['target_z']:.2f})",
            f"tilt     {m['tilt']:4.1f}°",
            f"pose err {m['pose_err']:.2f} rad",
            f"hold     {held:4.2f} / {m['hold_need']:.1f} s",
        ]
        h = 20 * len(lines) + 16
        d.rectangle([0, img.height - h, 340, img.height], fill=(12, 14, 18, 190))
        for i, line in enumerate(lines):
            d.text((16, img.height - h + 8 + 20 * i), line, font=self.font,
                   fill=(235, 235, 235))
        if done:
            d.text((img.width - 150, 104), "STOOD UP", font=self.title_font,
                   fill=(120, 230, 140))
        self.frames.append(np.asarray(img))


def title_card(text: str, subtitle: str = "", n_frames: int = 30) -> list[np.ndarray]:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (VIDEO_W, VIDEO_H), (10, 12, 16))
    d = ImageDraw.Draw(img)
    f1, f2 = make_font(40), make_font(22)
    w = d.textlength(text, font=f1)
    d.text(((VIDEO_W - w) / 2, VIDEO_H / 2 - 40), text, font=f1, fill=(240, 240, 240))
    if subtitle:
        w2 = d.textlength(subtitle, font=f2)
        d.text(((VIDEO_W - w2) / 2, VIDEO_H / 2 + 16), subtitle, font=f2,
               fill=(150, 180, 220))
    return [np.asarray(img)] * n_frames


# ---------------------------------------------------------------------------
# episode


def run_episode(sim: GetupSim, policy: HolosomaPolicy, term: Terminal, qpos: np.ndarray,
                duration_s: float, renderer=None, camera=None, overlay=None) -> dict:
    sim.reset_fallen(qpos)
    dt = sim.control_dt
    n = int(round(duration_s / dt))
    render_every = max(1, int(round((1.0 / VIDEO_FPS) / dt))) if renderer else 0

    prev_action = np.zeros(policy.n_dof)
    cmd = np.zeros(3)          # the get-up actor observes no command; kept for the API
    held = 0.0
    success_t = None
    peak_z = -np.inf

    for step in range(n):
        t = step * dt
        obs = sim.observe(prev_action, step, cmd)
        # Through the env's own action map (authority x wrist freeze). The
        # applied action is also what feeds back into the `actions` observation
        # — see A3Sim.apply_action for how that was verified.
        action = sim.apply_action(policy.act(obs))
        prev_action = action

        target = policy.default_dof_pos + policy.action_scale * np.clip(
            action, -ACTION_CLIP_VALUE, ACTION_CLIP_VALUE
        )
        for _ in range(sim.decimation):
            q = sim.data.qpos[sim.qpos_adr]
            qd = sim.data.qvel[sim.qvel_adr]
            tau = np.clip(
                policy.kp * (target - q) - policy.kd * qd,
                -sim.effort_limit, sim.effort_limit,
            )
            sim.data.ctrl[sim.act_ids] = tau
            mujoco.mj_step(sim.model, sim.data)

        if not np.isfinite(sim.data.qpos).all():
            break

        ok, m = term.check(sim)
        held = held + dt if ok else 0.0
        peak_z = max(peak_z, m["z"])
        if success_t is None and held >= term.hold_time:
            success_t = t

        if renderer is not None and step % render_every == 0:
            camera.lookat[:] = sim.base_pos()
            m2 = dict(m, tilt=math.degrees(math.acos(
                np.clip(-sim.projected_gravity()[2], -1.0, 1.0))),
                target_z=term.height, hold_need=term.hold_time)
            overlay(sim._render(renderer, camera), t, m2, held, success_t is not None)

    return {
        "success": success_t is not None,
        "success_t": success_t,
        "final_z": float(sim.base_pos()[2]),
        "peak_z": float(peak_z),
        "final_tilt_deg": math.degrees(math.acos(
            np.clip(-sim.projected_gravity()[2], -1.0, 1.0))),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--postures", default="supine,prone,side_left,side_right",
                   help="comma-separated; the pose bank's categories")
    p.add_argument("--episodes", type=int, default=3, help="episodes per posture")
    p.add_argument("--duration", type=float, default=10.0, help="episode seconds")
    p.add_argument("--name", default="getup")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-authority", type=float, default=None,
                   help="replay actions at this authority (beta). Default: the value "
                        "the checkpoint was trained at, read from its .pt. Pass 1.0 to "
                        "grade deployability at the manifest action scale.")
    args = p.parse_args()

    manifest = load_manifest()
    policy = HolosomaPolicy(args.onnx)
    term = Terminal(manifest, policy)
    amap = resolve_action_map(policy, manifest, args.action_authority)
    label = f"iter {policy.iteration}" if policy.iteration >= 0 else Path(args.onnx).stem
    label += f" · beta {amap['action_authority']:.2f} · unassisted"
    record = not args.no_video
    rng = np.random.default_rng(args.seed)

    print(f"policy   {args.onnx}")
    print(f"obs      {policy.obs_dim} dims  (layout: {policy.layout.source})")
    print(f"terms    {', '.join(t.name for t in policy.layout.terms)}")
    print(f"action   authority {amap['action_authority']:.2f} "
          f"[{amap['authority_source']}], wrists x{amap['wrist_action_factor']:.2f} "
          f"-> effective scale {policy.action_scale.mean() * amap['action_authority']:.3f}")
    if amap["action_authority"] != 1.0:
        print("         NOT the deploy scale — this grades the policy in distribution, "
              "not its deployability")
    if amap["trained_assist_scale"]:
        print(f"assist   trained with assist_scale {amap['trained_assist_scale']:.2f} "
              f"({amap['trained_assist_scale'] * 350:.0f} N torso pull); "
              "this rollout is UNASSISTED")
    print(f"gate     hold {term.hold_time:.0f}s at {term.height:.2f} +/- {term.height_tol:.2f} m, "
          f"pose <= {term.pose_tol} rad, |v| <= {term.lin_vel_max} m/s\n")

    model = build_model(None)
    if record:
        model.vis.global_.offwidth, model.vis.global_.offheight = VIDEO_W, VIDEO_H
    sim = GetupSim(policy, manifest, model=model)
    sim.action_authority = amap["action_authority"]
    sim.wrist_action_factor = amap["wrist_action_factor"]

    renderer = camera = None
    if record:
        renderer = mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.elevation, camera.azimuth = 4.0, -12.0, 130.0

    montage: list[np.ndarray] = []
    rows = []
    for posture in [s.strip() for s in args.postures.split(",") if s.strip()]:
        bank = load_poses(posture)
        picks = rng.choice(len(bank), size=min(args.episodes, len(bank)), replace=False)
        if record:
            montage += title_card(posture.replace("_", " ").upper(),
                                  f"{len(bank)} settled poses in the bank")
        for k, idx in enumerate(picks):
            ov = Overlay(f"{posture.replace('_', ' ')} — episode {k + 1}",
                         f"fallen pose #{idx} · policy must reach and hold the handoff pose",
                         label) if record else None
            r = run_episode(sim, policy, term, bank[idx], args.duration,
                            renderer, camera, ov)
            r |= {"posture": posture, "pose_idx": int(idx)}
            rows.append(r)
            flag = "STOOD UP" if r["success"] else "failed"
            print(f"  {posture:<11} #{idx:<4} {flag:<9} "
                  f"peak pelvis {r['peak_z']:.2f} m  final {r['final_z']:.2f} m  "
                  f"tilt {r['final_tilt_deg']:5.1f} deg")
            if ov:
                montage += ov.frames
                montage += [ov.frames[-1]] * (VIDEO_FPS // 2)

    if renderer is not None:
        renderer.close()

    n_ok = sum(r["success"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} episodes reached and held the handoff pose")
    by = {}
    for r in rows:
        by.setdefault(r["posture"], []).append(r["success"])
    for posture, v in by.items():
        print(f"  {posture:<11} {sum(v)}/{len(v)}")

    out = REPO_ROOT / "results" / "getup"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps({"onnx": args.onnx, "iteration": policy.iteration,
                    "action_map": amap, "assisted": False,
                    "episodes": rows, "success": n_ok,
                    "total": len(rows)}, indent=2), encoding="utf-8")
    print(f"\nmetrics  {out / f'{args.name}.json'}")

    if record and montage:
        import imageio.v2 as imageio

        vid_dir = REPO_ROOT / "results" / "videos" / args.name
        vid_dir.mkdir(parents=True, exist_ok=True)
        path = vid_dir / f"{args.name}.mp4"
        imageio.mimsave(path, montage, fps=VIDEO_FPS, quality=8, macro_block_size=1)
        print(f"video    {path}  ({len(montage) / VIDEO_FPS:.0f} s)")


if __name__ == "__main__":
    main()
