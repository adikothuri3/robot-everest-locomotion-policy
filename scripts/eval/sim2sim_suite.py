"""Sim2sim evaluation of a trained A3 Ultra locomotion policy in MuJoCo classic.

The policy trains on IsaacSim/PhysX in the cloud; this is the independent-physics
gate and the source of the demo videos. Observation/action contract and the
verification trail: `everest_locomotion.evaluation.sim2sim`.

Modes
  --mode sweep    screen every checkpoint in a run directory, rank, pick a winner
  --mode showcase rich scenario set (flat / rough / slope / friction / pushes /
                  alpine combo). With --video it writes ONE montage mp4 per run
                  covering the HIGHLIGHTS cut; --per-scenario-videos opts back in
                  to a file per scenario
  --mode grid     the 68-scenario stability grid from notes/baselines.md, run for
                  the policy and (with --with-baseline) for the PD-stand control
                  on the same asset, so the comparison is head-to-head

Examples
  python scripts/eval/sim2sim_suite.py --mode sweep --run-dir checkpoints/cloud_...
  python scripts/eval/sim2sim_suite.py --mode showcase --onnx <ckpt>.onnx --video
  python scripts/eval/sim2sim_suite.py --mode grid --onnx <ckpt>.onnx --with-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402

from everest_locomotion.robots.manifest import load_manifest  # noqa: E402
from everest_locomotion.terrains import TerrainSpec, procedural_rough  # noqa: E402
from everest_locomotion.evaluation.sim2sim import (  # noqa: E402
    CONTROL_DT,
    A3Sim,
    Command,
    HolosomaPolicy,
    Push,
    build_model,
)

REPO = Path(__file__).resolve().parents[2]
# 25 fps divides the 50 Hz control rate exactly (render every 2nd step), so playback
# is real time. A rate that does not divide it silently speeds the video up.
VIDEO_W, VIDEO_H, VIDEO_FPS = 960, 540, 25

GROUP_BLURB = {
    "flat": "flat ground — the full command envelope",
    "rough": "procedural rough terrain",
    "slope": "graded slopes (never seen in training)",
    "friction": "reduced ground friction",
    "push": "external shoves while walking",
    "payload": "added payload mass",
    "alpine": "everything at once — where it breaks",
}


# ---------------------------------------------------------------------------
# scenarios


@dataclass
class Scenario:
    name: str
    group: str
    duration_s: float = 10.0
    command: Command = field(default_factory=Command)
    pushes: list[Push] = field(default_factory=list)
    terrain: TerrainSpec | None = None
    friction: float | None = None
    added_mass_kg: float = 0.0
    spawn_yaw: float = 0.0
    caption: str = ""
    highlight: bool = False   # include in the montage cut


def rough(
    difficulty: float,
    slope_deg: float = 0.0,
    seed: int = 7,
    roughness: float = 0.08,
    boxes: int = 60,
    box_height_m: float = 0.10,
):
    """Undulating noise + discrete steps.

    Training terrain was 60% rough / 20% low obstacles / 20% flat with 1–5 cm
    amplitude, so difficulty 1.0 here (8 cm noise + 10 cm steps) is beyond what
    the policy saw.
    """
    return TerrainSpec(
        seed=seed,
        difficulty=difficulty,
        slope_deg=slope_deg,
        size_m=(24.0, 24.0),
        resolution_m=0.05,
        roughness_m=roughness,
        obstacle_params={"n_boxes": boxes, "box_height_m": box_height_m, "box_size_m": 0.6},
    )


#: Scenarios that make the montage cut — one per capability, plus the failures.
HIGHLIGHTS = {
    "flat_command_sequence", "flat_walk_fwd_1.0", "flat_turn_in_place_1.0",
    "flat_strafe_left_0.5", "rough_d1", "slope_up_10deg", "slope_down_15deg",
    "friction_mu0.2", "push_front_2", "push_right_2", "payload_10kg",
    "alpine_combo",
}


def showcase_scenarios() -> list[Scenario]:
    s: list[Scenario] = []

    # --- flat ground, the command envelope the policy was trained on ---------
    s += [
        Scenario("flat_stand", "flat", 8.0, Command(), caption="zero command — quiet stand"),
        Scenario("flat_walk_fwd_0.5", "flat", 10.0, Command(lin_vel_x=0.5),
                 caption="forward 0.5 m/s"),
        Scenario("flat_walk_fwd_1.0", "flat", 10.0, Command(lin_vel_x=1.0),
                 caption="forward 1.0 m/s (command limit)"),
        Scenario("flat_walk_back_0.5", "flat", 10.0, Command(lin_vel_x=-0.5),
                 caption="backward 0.5 m/s"),
        Scenario("flat_strafe_left_0.5", "flat", 10.0, Command(lin_vel_y=0.5),
                 caption="strafe left 0.5 m/s"),
        Scenario("flat_strafe_right_0.5", "flat", 10.0, Command(lin_vel_y=-0.5),
                 caption="strafe right 0.5 m/s"),
        Scenario("flat_turn_in_place_1.0", "flat", 10.0, Command(ang_vel_yaw=1.0),
                 caption="turn in place 1.0 rad/s"),
        Scenario("flat_diagonal", "flat", 10.0, Command(lin_vel_x=0.6, lin_vel_y=0.4),
                 caption="diagonal 0.6 / 0.4 m/s"),
        Scenario("flat_arc", "flat", 12.0, Command(lin_vel_x=0.7, ang_vel_yaw=0.5),
                 caption="walk + turn (arc)"),
        Scenario(
            "flat_command_sequence", "flat", 24.0,
            Command(schedule=[
                (0.0, (0.0, 0.0, 0.0)),
                (3.0, (0.8, 0.0, 0.0)),
                (8.0, (0.8, 0.0, 0.8)),
                (12.0, (0.0, 0.6, 0.0)),
                (16.0, (-0.6, 0.0, 0.0)),
                (20.0, (0.0, 0.0, 0.0)),
            ]),
            caption="live command switching: stand → walk → arc → strafe → back → stand",
        ),
    ]

    # --- rough terrain ------------------------------------------------------
    for d in (0.25, 0.5, 0.75, 1.0):
        s.append(Scenario(
            f"rough_d{d:g}", "rough", 12.0, Command(lin_vel_x=0.5), terrain=rough(d),
            caption=(f"rough terrain, difficulty {d:g} — "
                     f"±{0.08 * d * 100:.0f} cm noise, steps to {10 * d:.0f} cm"),
        ))

    # --- slopes (downhill runs spawn facing downhill) ------------------------
    for slope in (5.0, 10.0, 15.0):
        s.append(Scenario(
            f"slope_up_{slope:g}deg", "slope", 12.0, Command(lin_vel_x=0.5),
            terrain=rough(0.25, slope_deg=slope, boxes=20), caption=f"{slope:g}° climb",
        ))
        s.append(Scenario(
            f"slope_down_{slope:g}deg", "slope", 12.0, Command(lin_vel_x=0.5),
            terrain=rough(0.25, slope_deg=slope, boxes=20), spawn_yaw=np.pi,
            caption=f"{slope:g}° descent",
        ))

    # --- friction -----------------------------------------------------------
    for mu in (0.5, 0.3, 0.2, 0.1):
        s.append(Scenario(
            f"friction_mu{mu:g}", "friction", 10.0, Command(lin_vel_x=0.5), friction=mu,
            caption=f"low friction ground, mu = {mu:g}",
        ))

    # --- pushes -------------------------------------------------------------
    for label, vec in (
        ("front", (1.0, 0.0)), ("back", (-1.0, 0.0)),
        ("left", (0.0, 1.0)), ("right", (0.0, -1.0)),
    ):
        for mag in (1.0, 1.5, 2.0):
            s.append(Scenario(
                f"push_{label}_{mag:g}", "push", 9.0, Command(lin_vel_x=0.5),
                pushes=[Push(3.0, (vec[0] * mag, vec[1] * mag, 0.0)),
                        Push(6.0, (vec[0] * mag, vec[1] * mag, 0.0))],
                caption=f"{mag:g} m/s shove from the {label}, twice, while walking",
            ))

    # --- payload ------------------------------------------------------------
    for kg in (5.0, 10.0):
        s.append(Scenario(
            f"payload_{kg:g}kg", "payload", 10.0, Command(lin_vel_x=0.5), added_mass_kg=kg,
            caption=f"+{kg:g} kg carried on the pelvis ({kg / 60.2 * 100:.0f}% of body mass)",
        ))

    # --- everything at once -------------------------------------------------
    s += [
        Scenario(
            "alpine_combo", "alpine", 14.0, Command(lin_vel_x=0.5),
            terrain=rough(0.8, slope_deg=10.0, roughness=0.09), friction=0.4,
            pushes=[Push(5.0, (0.0, 1.2, 0.0)), Push(9.0, (-1.2, 0.0, 0.0))],
            caption="alpine proxy: rough d0.8 + 10° climb + mu 0.4 + gusts",
        ),
        Scenario(
            "alpine_combo_hard", "alpine", 14.0, Command(lin_vel_x=0.5),
            terrain=rough(1.0, slope_deg=15.0, roughness=0.10), friction=0.3,
            pushes=[Push(4.0, (0.0, 1.5, 0.0)), Push(8.0, (-1.5, 0.0, 0.0))],
            caption="alpine proxy, hard: rough d1.0 + 15° climb + mu 0.3 + gusts",
        ),
        Scenario(
            "alpine_descent", "alpine", 14.0, Command(lin_vel_x=0.5),
            terrain=rough(0.8, slope_deg=12.0, roughness=0.09), friction=0.4,
            spawn_yaw=np.pi, pushes=[Push(6.0, (0.0, -1.2, 0.0))],
            caption="alpine proxy: 12° descent on rough d0.8 + mu 0.4 + gust",
        ),
    ]
    for sc in s:
        sc.highlight = sc.name in HIGHLIGHTS
    return s


def grid_scenarios() -> list[Scenario]:
    """The 68-scenario stability grid recorded for the PD-stand floor.

    Same families and same parameter values as `scripts/eval/stability_suite.py`
    (quiet stand, 8x6 push grid, 5x3 terrain grid, 4 frictions), so the survival
    count is directly comparable to `results/stability/pd_stand_baseline.json`.
    """
    s = [Scenario("quiet_stand", "quiet_stand", 10.0, Command())]

    dirs = {
        "front": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1),
        "fl": (0.707, 0.707), "fr": (0.707, -0.707),
        "bl": (-0.707, 0.707), "br": (-0.707, -0.707),
    }
    for name, (dx, dy) in dirs.items():
        for mag in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            s.append(Scenario(
                f"push_{name}_{mag:g}", "push", 6.0, Command(),
                pushes=[Push(2.0, (dx * mag, dy * mag, 0.0))],
            ))

    for diff in (0.2, 0.4, 0.6, 0.8, 1.0):
        for slope in (0.0, 5.0, 10.0):
            s.append(Scenario(
                f"terrain_d{diff:g}_s{slope:g}", "terrain", 6.0, Command(),
                terrain=TerrainSpec(seed=0, difficulty=diff, slope_deg=slope,
                                    size_m=(10.0, 10.0), roughness_m=0.08),
            ))

    for mu in (0.7, 0.4, 0.25, 0.15):
        s.append(Scenario(
            f"friction_mu{mu:g}", "friction", 6.0, Command(), friction=mu,
            pushes=[Push(2.0, (0.1, 0.0, 0.0))],
        ))
    return s


# ---------------------------------------------------------------------------
# PD-stand control (same asset, same harness)


class StandPolicy:
    """Holds the training default pose. The floor every trained policy must beat.

    Uses the diagnostics' stiff hold gains: RL gains cannot passively hold a
    60 kg inverted pendulum, so the honest floor gives the baseline its best
    shot (see `scripts/diagnostics/check_mujoco_model.py`).
    """

    HOLD_MIN = {"ankle": (800.0, 20.0), "hip": (400.0, 10.0),
                "knee": (400.0, 10.0), "waist": (400.0, 10.0)}

    def __init__(self, ref: HolosomaPolicy):
        self.dof_names = list(ref.dof_names)
        self.default_dof_pos = ref.default_dof_pos.copy()
        self.action_scale = ref.action_scale.copy()
        self.n_dof = ref.n_dof
        self.obs_dim = ref.obs_dim
        self.iteration = -1
        # A3Sim assembles observations from the policy's layout even for a policy
        # that ignores them: borrowing the reference policy's layout keeps the
        # baseline on the identical control period and observation pipeline, so
        # the head-to-head comparison differs only in the action.
        self.layout = ref.layout
        self.control_dt = ref.control_dt
        self.kp, self.kd = ref.kp.copy(), ref.kd.copy()
        for i, name in enumerate(self.dof_names):
            for key, (kp, kd) in self.HOLD_MIN.items():
                if key in name:
                    self.kp[i] = max(self.kp[i], kp)
                    self.kd[i] = max(self.kd[i], kd)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_dof)  # action 0 == hold the default pose


# ---------------------------------------------------------------------------
# rendering


def make_font(size: int):
    from PIL import ImageFont

    for name in ("consola.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Overlay:
    def __init__(self, scenario: Scenario, policy_label: str):
        from PIL import ImageDraw, ImageFont  # noqa: F401

        self.scenario = scenario
        self.policy_label = policy_label
        self.title_font = make_font(24)
        self.font = make_font(18)
        self.small = make_font(15)
        self.frames: list[np.ndarray] = []

    def __call__(self, frame, t, cmd, v_body, w_body, tilt, height, alive):
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, img.width, 92], fill=(12, 14, 18, 190))
        d.text((16, 10), self.scenario.name.replace("_", " "), font=self.title_font,
               fill=(245, 245, 245))
        d.text((16, 42), self.scenario.caption, font=self.small, fill=(170, 200, 235))
        d.text((16, 64), f"A3 Ultra · FastSAC · {self.policy_label} · MuJoCo (sim2sim)",
               font=self.small, fill=(140, 150, 165))

        lines = [
            f"t        {t:5.2f} s",
            f"cmd      vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
            f"actual   vx {v_body[0]:+.2f}  vy {v_body[1]:+.2f}  wz {w_body[2]:+.2f}",
            f"pelvis   {height:.2f} m",
            f"tilt     {tilt:4.1f}°",
        ]
        h = 20 * len(lines) + 16
        d.rectangle([0, img.height - h, 330, img.height], fill=(12, 14, 18, 190))
        for i, line in enumerate(lines):
            d.text((16, img.height - h + 8 + 20 * i), line, font=self.font,
                   fill=(235, 235, 235))
        if not alive:
            d.text((img.width // 2 - 60, img.height // 2), "FALL", font=self.title_font,
                   fill=(255, 90, 90))
        self.frames.append(np.asarray(img))


def title_card(text: str, subtitle: str = "", n_frames: int = 24) -> list[np.ndarray]:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (VIDEO_W, VIDEO_H), (10, 12, 16))
    d = ImageDraw.Draw(img)
    f1, f2 = make_font(40), make_font(22)
    w = d.textlength(text, font=f1)
    d.text(((VIDEO_W - w) / 2, VIDEO_H / 2 - 40), text, font=f1, fill=(240, 240, 240))
    if subtitle:
        w2 = d.textlength(subtitle, font=f2)
        d.text(((VIDEO_W - w2) / 2, VIDEO_H / 2 + 16), subtitle, font=f2, fill=(150, 180, 220))
    return [np.asarray(img)] * n_frames


# ---------------------------------------------------------------------------
# runner


def run_scenario(policy, manifest, sc: Scenario, record: bool, policy_label: str,
                 seed: int = 0):
    patch = procedural_rough(sc.terrain) if sc.terrain is not None else None
    model = build_model(patch)
    if record:
        model.vis.global_.offwidth = VIDEO_W
        model.vis.global_.offheight = VIDEO_H
    sim = A3Sim(policy, manifest, model=model, patch=patch, friction=sc.friction,
                added_base_mass_kg=sc.added_mass_kg)

    renderer = camera = overlay = None
    if record:
        renderer = mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.6
        camera.azimuth = 135.0 + np.degrees(sc.spawn_yaw)
        camera.elevation = -10.0
        overlay = Overlay(sc, policy_label)

    result = sim.run(
        duration_s=sc.duration_s,
        command=sc.command,
        pushes=sc.pushes,
        renderer=renderer,
        camera=camera,
        video_fps=VIDEO_FPS,
        frame_cb=overlay,
        name=sc.name,
        seed=seed,
        spawn_yaw=sc.spawn_yaw,
    )
    if renderer is not None:
        renderer.close()
    frames = overlay.frames if overlay else []
    return result, frames


def write_video(path: Path, frames: list[np.ndarray], fps: int = VIDEO_FPS) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=1)


# ---------------------------------------------------------------------------
# modes


def mode_sweep(args, manifest) -> None:
    run_dir = Path(args.run_dir)
    ckpts = sorted(run_dir.glob("model_*.onnx"))
    if not ckpts:
        raise SystemExit(f"no ONNX checkpoints in {run_dir}")

    screen = [
        Scenario("stand", "flat", 6.0, Command()),
        Scenario("fwd_0.5", "flat", 8.0, Command(lin_vel_x=0.5)),
        Scenario("fwd_1.0", "flat", 8.0, Command(lin_vel_x=1.0)),
        Scenario("strafe_0.5", "flat", 8.0, Command(lin_vel_y=0.5)),
        Scenario("turn_1.0", "flat", 8.0, Command(ang_vel_yaw=1.0)),
        Scenario("rough_d0.6", "rough", 8.0, Command(lin_vel_x=0.5), terrain=rough(0.6)),
        Scenario("push_1.5", "push", 8.0, Command(lin_vel_x=0.5),
                 pushes=[Push(3.0, (-1.5, 0.0, 0.0)), Push(5.5, (0.0, 1.5, 0.0))]),
    ]

    rows = []
    for ck in ckpts:
        policy = HolosomaPolicy(ck)
        t0 = time.time()
        results = [run_scenario(policy, manifest, sc, False, ck.stem)[0] for sc in screen]
        survived = sum(r.survived for r in results)
        track = [r for r in results if np.isfinite(r.lin_vel_error)]
        row = {
            "checkpoint": ck.name,
            "iteration": policy.iteration,
            "survived": survived,
            "n": len(results),
            "lin_vel_error": float(np.mean([r.lin_vel_error for r in track])),
            "ang_vel_error": float(np.mean([r.ang_vel_error for r in track])),
            "action_jitter": float(np.mean([r.action_jitter for r in results])),
            "max_tilt_deg": float(np.max([r.max_tilt_deg for r in results])),
            "mean_base_height": float(np.mean([r.mean_base_height for r in results])),
            "seconds": round(time.time() - t0, 1),
            "per_scenario": {r.name: {"survived": r.survived,
                                      "lin_vel_error": r.lin_vel_error,
                                      "fall_time_s": r.fall_time_s} for r in results},
        }
        rows.append(row)
        print(f"{ck.name:22s} it={policy.iteration:6d} survived {survived}/{len(results)} "
              f"velerr={row['lin_vel_error']:.3f} angerr={row['ang_vel_error']:.3f} "
              f"jitter={row['action_jitter']:.3f} ({row['seconds']}s)")

    rows.sort(key=lambda r: (-r["survived"], r["lin_vel_error"] + 0.5 * r["ang_vel_error"]))
    best = rows[0]
    out = REPO / "results" / "sim2sim" / "checkpoint_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_dir": str(run_dir), "ranked": rows}, indent=2))
    print(f"\nbest: {best['checkpoint']} (survived {best['survived']}/{best['n']}, "
          f"lin_vel_error {best['lin_vel_error']:.3f})")
    print(f"results: {out}")


def mode_showcase(args, manifest) -> None:
    policy = HolosomaPolicy(args.onnx)
    label = f"iter {policy.iteration:,}"
    scenarios = showcase_scenarios()
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.name in wanted or s.group in wanted]

    vid_dir = REPO / "results" / "videos" / args.name
    rows, montage = [], []
    group_seen: set[str] = set()

    for sc in scenarios:
        t0 = time.time()
        # Render only what reaches the montage. A stage produces ONE video; 41
        # separate mp4s per stage was ~200 MB of files nobody opens, and the
        # off-screen rendering dominated the run time for scenarios whose frames
        # were then discarded. `--per-scenario-videos` brings the old files back
        # when a single case actually needs to be inspected on its own.
        want_frames = bool(args.video) and (sc.highlight or args.per_scenario_videos)
        result, frames = run_scenario(policy, manifest, sc, want_frames, label)
        rows.append({"group": sc.group, "caption": sc.caption, **result.as_row()})
        status = "OK  " if result.survived else "FALL"
        print(f"[{status}] {sc.name:26s} velerr={result.lin_vel_error:6.3f} "
              f"speed={result.mean_speed_ms:5.2f} tilt={result.max_tilt_deg:5.1f} "
              f"h={result.mean_base_height:.2f} jit={result.action_jitter:.3f} "
              f"({time.time() - t0:4.1f}s)")

        if frames:
            if not result.survived:
                frames += [frames[-1]] * VIDEO_FPS  # hold the FALL frame for a second
            if args.per_scenario_videos:
                write_video(vid_dir / f"{sc.name}.mp4", frames)
            if sc.highlight:
                if sc.group not in group_seen:
                    group_seen.add(sc.group)
                    montage += title_card(sc.group.upper(), GROUP_BLURB.get(sc.group, ""))
                montage += title_card(sc.name, sc.caption, n_frames=18)
                montage += frames

    n_ok = sum(r["survived"] for r in rows)
    summary = {
        "policy": str(Path(args.onnx)),
        "iteration": policy.iteration,
        "n_scenarios": len(rows),
        "n_survived": n_ok,
        "survival_rate": n_ok / len(rows),
        "by_group": {},
    }
    for g in sorted({r["group"] for r in rows}):
        gr = [r for r in rows if r["group"] == g]
        track = [r["lin_vel_error"] for r in gr if np.isfinite(r["lin_vel_error"])]
        summary["by_group"][g] = {
            "n": len(gr),
            "survived": sum(r["survived"] for r in gr),
            "mean_lin_vel_error": float(np.mean(track)) if track else None,
            "mean_action_jitter": float(np.mean([r["action_jitter"] for r in gr])),
            "max_tilt_deg": float(np.max([r["max_tilt_deg"] for r in gr])),
        }

    out = REPO / "results" / "sim2sim" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nsurvived {n_ok}/{len(rows)}   results: {out}")

    if args.video and montage:
        montage_path = vid_dir / "00_montage.mp4"
        montage = title_card("A3 Ultra — learned locomotion",
                             f"FastSAC @ {policy.iteration:,} iterations · MuJoCo sim2sim",
                             n_frames=60) + montage
        write_video(montage_path, montage)
        print(f"montage: {montage_path}  ({len(montage)} frames, "
              f"{len(montage) / VIDEO_FPS:.0f}s)")


def mode_grid(args, manifest) -> None:
    policy = HolosomaPolicy(args.onnx)
    runs = [("policy", policy)]
    if args.with_baseline:
        runs.append(("pd_stand", StandPolicy(policy)))

    out_all = {}
    for label, pol in runs:
        rows = []
        t0 = time.time()
        for sc in grid_scenarios():
            result, _ = run_scenario(pol, manifest, sc, False, label)
            rows.append({"group": sc.group, **result.as_row()})
        n_ok = sum(r["survived"] for r in rows)

        max_push = {}
        for r in rows:
            if r["group"] == "push" and r["survived"]:
                parts = r["name"].split("_")
                d, mag = parts[1], float(parts[2])
                max_push[d] = max(max_push.get(d, 0.0), mag)
        worst_terrain = [r["name"] for r in rows if r["group"] == "terrain" and not r["survived"]]
        worst_friction = [r["name"] for r in rows if r["group"] == "friction" and not r["survived"]]

        out_all[label] = {
            "n_scenarios": len(rows),
            "n_survived": n_ok,
            "survival_rate": n_ok / len(rows),
            "max_recoverable_push_ms": max_push,
            "terrain_failures": worst_terrain,
            "friction_failures": worst_friction,
            "seconds": round(time.time() - t0, 1),
            "rows": rows,
        }
        print(f"{label:9s} survived {n_ok}/{len(rows)}  max_push={max_push}")

    out = REPO / "results" / "sim2sim" / f"{args.name}_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_all, indent=2))
    print(f"results: {out}")


def mode_pushlimit(args, manifest) -> None:
    """Escalate impulse magnitude per direction until the policy falls.

    The `grid` mode tops out at 1.0 m/s (the floor's grid), which the trained
    policy clears everywhere, so the ceiling has to be found separately.
    """
    policy = HolosomaPolicy(args.onnx)
    dirs = {
        "front": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1),
        "fl": (0.707, 0.707), "fr": (0.707, -0.707),
        "bl": (-0.707, 0.707), "br": (-0.707, -0.707),
    }
    mags = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    out = {}
    for stance, cmd in (("standing", Command()), ("walking", Command(lin_vel_x=0.5))):
        limits = {}
        for name, (dx, dy) in dirs.items():
            best = 0.0
            for mag in mags:
                sc = Scenario(
                    f"push_{name}_{mag:g}", "push", 6.0, cmd,
                    pushes=[Push(2.0, (dx * mag, dy * mag, 0.0))],
                )
                if not run_scenario(policy, manifest, sc, False, "limit")[0].survived:
                    break
                best = mag
            limits[name] = best
            print(f"{stance:9s} {name:6s} max recoverable {best:.1f} m/s")
        out[stance] = limits
    path = REPO / "results" / "sim2sim" / f"{args.name}_pushlimit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"results: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sweep", "showcase", "grid", "pushlimit"],
                   default="showcase")
    p.add_argument("--onnx", help="exported policy (showcase/grid)")
    p.add_argument("--run-dir", help="checkpoint directory (sweep)")
    p.add_argument("--name", default="sim2sim")
    p.add_argument("--video", action="store_true",
                   help="render ONE montage mp4 covering the highlight scenarios")
    p.add_argument("--per-scenario-videos", action="store_true",
                   help="also write a separate mp4 per scenario (~40 files, ~200 MB)")
    p.add_argument("--only", help="comma-separated scenario or group names")
    p.add_argument("--with-baseline", action="store_true",
                   help="also run the PD-stand control through the same grid")
    args = p.parse_args()

    manifest = load_manifest()
    if args.mode == "sweep":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --mode sweep")
        mode_sweep(args, manifest)
    elif args.mode == "showcase":
        if not args.onnx:
            raise SystemExit("--onnx is required for --mode showcase")
        mode_showcase(args, manifest)
    elif args.mode == "pushlimit":
        if not args.onnx:
            raise SystemExit("--onnx is required for --mode pushlimit")
        mode_pushlimit(args, manifest)
    else:
        if not args.onnx:
            raise SystemExit("--onnx is required for --mode grid")
        mode_grid(args, manifest)


if __name__ == "__main__":
    main()
