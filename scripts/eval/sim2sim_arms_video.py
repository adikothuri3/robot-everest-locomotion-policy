"""Video evidence for the arm-motion question (E08b/E08c).

Renders two things:

1. Side-by-side clips — identical arm motion, identical physics, the only
   difference being whether the locomotion policy is allowed to *see* the moved
   arms. Left panel sees them (today's behaviour), right panel has the 14 arm
   observation channels frozen at default.
2. Solo clips of the frozen-observation policy walking through progressively
   nastier upper-body motion, including hand payloads and rough terrain.

    python scripts/eval/sim2sim_arms_video.py --onnx <policy>.onnx
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402

from everest_locomotion.robots.manifest import load_manifest  # noqa: E402
from everest_locomotion.terrains import procedural_rough  # noqa: E402
from everest_locomotion.evaluation.sim2sim import (  # noqa: E402
    A3Sim, Command, HolosomaPolicy, Push, build_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim2sim_arms import ARM_JOINTS, ArmSkill, both, const, mask_arm_obs, rough  # noqa: E402
from sim2sim_suite import make_font, title_card, write_video  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FPS = 25
PANEL_W, PANEL_H = 660, 560
SOLO_W, SOLO_H = 960, 540
OUT = REPO / "results" / "videos" / "arms"


# ---------------------------------------------------------------------------
# arm motions


def _rand_pose_fn(manifest, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([manifest.joints[n]["limits_rad"][0] for n in ARM_JOINTS])
    hi = np.array([manifest.joints[n]["limits_rad"][1] for n in ARM_JOINTS])
    steps = rng.uniform(0.15, 0.85, size=(40, 14))
    return {n: (lambda t, k=k: lo[k] + steps[int(t / 0.5) % 40][k] * (hi[k] - lo[k]))
            for k, n in enumerate(ARM_JOINTS)}


def skills(manifest):
    return {
        "reach_forward": ArmSkill(
            "reach_forward", "both arms reached out in front and held",
            {**both("shoulder_pitch_joint", const(-1.4)),
             **both("elbow_joint", const(0.9))}),
        "raise_overhead": ArmSkill(
            "raise_overhead", "both arms raised overhead",
            both("shoulder_pitch_joint", const(-2.6))),
        "swing_3hz": ArmSkill(
            "swing_3hz", "arms swinging antiphase at 3 Hz, +/-1.2 rad",
            {"left_shoulder_pitch_joint":
             lambda t: -1.2 * math.sin(2 * math.pi * 3.0 * t),
             "right_shoulder_pitch_joint":
             lambda t: 1.2 * math.sin(2 * math.pi * 3.0 * t)}, ramp_s=0.5),
        "slam_extremes": ArmSkill(
            "slam_extremes", "slamming between joint extremes every 0.5 s",
            {**both("shoulder_pitch_joint", lambda t: -2.6 if int(t / 0.5) % 2 else 0.3),
             **both("elbow_joint", lambda t: 1.6 if int(t / 0.5) % 2 else 0.0)},
            ramp_s=0.15),
        "random_full_range": ArmSkill(
            "random_full_range", "every arm joint resampled across its full range, 2 Hz",
            _rand_pose_fn(manifest), ramp_s=0.3),
    }


# ---------------------------------------------------------------------------
# overlay


class ArmOverlay:
    def __init__(self, sim, title, subtitle, note="", width=SOLO_W, compact=False):
        self.sim, self.title, self.subtitle, self.note = sim, title, subtitle, note
        self.compact = compact
        self.f_title = make_font(22 if compact else 25)
        self.f_sub = make_font(14 if compact else 16)
        self.f_data = make_font(15 if compact else 18)
        self.frames: list[np.ndarray] = []
        self.fell = False
        self._sh = [sim.policy.dof_names.index(n) for n in
                    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")]

    def __call__(self, frame, t, cmd, v_body, w_body, tilt, height, alive):
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        head_h = 66 if self.compact else 92
        d.rectangle([0, 0, img.width, head_h], fill=(12, 14, 18, 200))
        d.text((14, 8), self.title, font=self.f_title, fill=(245, 245, 245))
        d.text((14, 36 if self.compact else 42), self.subtitle, font=self.f_sub,
               fill=(150, 185, 225))
        if self.note and not self.compact:
            d.text((14, 66), self.note, font=self.f_sub, fill=(140, 150, 165))

        if not alive or tilt > 60.0 or height < 0.45:
            self.fell = True

        sh = self.sim.data.qpos[self.sim.qpos_adr][self._sh]
        lines = [
            f"t        {t:5.2f} s",
            f"cmd vx   {cmd[0]:+.2f}   actual {v_body[0]:+.2f}",
            f"yaw rate {w_body[2]:+.2f} rad/s",
            f"shoulder {sh[0]:+.2f} / {sh[1]:+.2f} rad",
            f"pelvis   {height:.2f} m    tilt {tilt:4.1f}",
        ]
        h = 20 * len(lines) + 16
        d.rectangle([0, img.height - h, 300 if self.compact else 340, img.height],
                    fill=(12, 14, 18, 200))
        for i, line in enumerate(lines):
            d.text((14, img.height - h + 8 + 20 * i), line, font=self.f_data,
                   fill=(232, 232, 232))
        status = "FALLEN" if self.fell else "WALKING"
        colour = (255, 100, 100) if self.fell else (120, 215, 165)
        d.text((img.width - 105, img.height - 32), status, font=self.f_data, fill=colour)
        self.frames.append(np.asarray(img))


# ---------------------------------------------------------------------------
# runners


def build(policy, manifest, terrain=None, hand_kg=0.0, w=SOLO_W, h=SOLO_H):
    patch = procedural_rough(rough(0.5)) if terrain else None
    model = build_model(patch)
    model.vis.global_.offwidth, model.vis.global_.offheight = w, h
    sim = A3Sim(policy, manifest, model=model, patch=patch)
    if hand_kg:
        for s in ("left", "right"):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_Link")
            model.body_mass[b] += hand_kg
    return sim, model


def camera(w, h, dist=3.5):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = dist, 128.0, -10.0
    return cam


def run_clip(policy, manifest, skill, cmd, *, masked, terrain=None, hand_kg=0.0,
             pushes=None, duration=12.0, title="", subtitle="", note="",
             w=SOLO_W, h=SOLO_H, compact=False, keep_filming=False):
    sim, model = build(policy, manifest, terrain, hand_kg, w, h)
    renderer = mujoco.Renderer(model, height=h, width=w)
    ov = ArmOverlay(sim, title, subtitle, note, compact=compact)
    kw = {"fall_tilt_deg": 179.0, "fall_height_m": 0.05} if keep_filming else {}
    res = sim.run(
        duration_s=duration, command=cmd, pushes=pushes or [],
        target_override=skill.make_override(sim) if skill else None,
        obs_transform=mask_arm_obs(sim) if masked else None,
        renderer=renderer, camera=camera(w, h), video_fps=FPS, frame_cb=ov, **kw,
    )
    renderer.close()
    return res, ov.frames


def side_by_side(left, right, label):
    from PIL import Image, ImageDraw

    n = max(len(left), len(right))
    left = left + [left[-1]] * (n - len(left))
    right = right + [right[-1]] * (n - len(right))
    f = make_font(17)
    out = []
    for a, b in zip(left, right):
        canvas = Image.new("RGB", (PANEL_W * 2 + 4, PANEL_H + 34), (10, 12, 16))
        canvas.paste(Image.fromarray(a), (0, 34))
        canvas.paste(Image.fromarray(b), (PANEL_W + 4, 34))
        d = ImageDraw.Draw(canvas)
        d.text(((canvas.width - d.textlength(label, font=f)) / 2, 8), label, font=f,
               fill=(190, 198, 210))
        out.append(np.asarray(canvas))
    return out


def card(text: str, subtitle: str, size: tuple[int, int], n: int = 50):
    """Title card sized to match the clips it is spliced between.

    The comparison clips (two panels) and the solo clips have different frame
    sizes, so they cannot share one montage — hence one montage each.
    """
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), (10, 12, 16))
    d = ImageDraw.Draw(img)
    f1, f2 = make_font(38), make_font(21)
    d.text(((w - d.textlength(text, font=f1)) / 2, h / 2 - 38), text, font=f1,
           fill=(240, 240, 240))
    if subtitle:
        d.text(((w - d.textlength(subtitle, font=f2)) / 2, h / 2 + 14), subtitle,
               font=f2, fill=(150, 180, 220))
    return [np.asarray(img)] * n


def assemble_montages(order_compare, order_solo) -> None:
    """Concatenate the already-written clips. Avoids re-simulating everything."""
    import imageio.v2 as imageio

    def read(path):
        r = imageio.get_reader(path)
        frames = [f for f in r]
        r.close()
        return frames

    for out_name, names, title, sub in (
        ("00_montage_compare.mp4", order_compare, "Does the policy SEE the arms?",
         "identical arm motion — only the observation differs"),
        ("01_montage_solo.mp4", order_solo, "Observation frozen",
         "the skill owns the arms; the policy walks blind to them"),
    ):
        paths = [OUT / f"{n}.mp4" for n in names]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        frames = read(paths[0])
        size = (frames[0].shape[1], frames[0].shape[0])
        montage = card(title, sub, size) + frames
        for p in paths[1:]:
            montage += read(p)
        write_video(OUT / out_name, montage, fps=FPS)
        print(f"{out_name}: {len(montage)} frames ({len(montage) / FPS:.0f}s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--montage-only", action="store_true",
                   help="rebuild montages from clips already on disk")
    args = p.parse_args()

    compare_order = ["00_compare_reach_forward", "00_compare_slam_extremes",
                     "00_compare_random_full_range"]
    solo_order = ["reach_forward", "raise_overhead", "swing_3hz", "slam_extremes",
                  "random_full_range", "hands_6kg", "rough_push"]
    if args.montage_only:
        assemble_montages(compare_order, solo_order)
        return

    manifest = load_manifest()
    policy = HolosomaPolicy(args.onnx)
    S = skills(manifest)

    # --- 1. the comparison: same arm motion, policy blind vs not --------------
    for key in ("reach_forward", "slam_extremes", "random_full_range"):
        sk = S[key]
        clips = []
        for masked in (False, True):
            _, frames = run_clip(
                policy, manifest, sk, Command(lin_vel_x=0.5), masked=masked,
                duration=12.0, w=PANEL_W, h=PANEL_H, compact=True, keep_filming=True,
                title="arms hidden from policy" if masked else "arms visible to policy",
                subtitle="14 arm obs channels frozen" if masked else "today's behaviour",
            )
            clips.append(frames)
        pair = side_by_side(clips[0], clips[1],
                            f"A3 Ultra · walking 0.5 m/s · skill drives the arms: {sk.description}")
        write_video(OUT / f"00_compare_{key}.mp4", pair, fps=FPS)
        print(f"compare {key}: {len(pair)} frames")

    # --- 2. solo clips, observation frozen -----------------------------------
    solo = [
        ("reach_forward", S["reach_forward"], Command(lin_vel_x=0.5), None, 0.0, None,
         "arms held out in front while walking"),
        ("raise_overhead", S["raise_overhead"], Command(lin_vel_x=0.5), None, 0.0, None,
         "arms overhead while walking"),
        ("swing_3hz", S["swing_3hz"], Command(lin_vel_x=1.0), None, 0.0, None,
         "3 Hz arm swing at full walking speed"),
        ("slam_extremes", S["slam_extremes"], Command(lin_vel_x=0.5), None, 0.0, None,
         "slamming between joint extremes"),
        ("random_full_range", S["random_full_range"], Command(lin_vel_x=0.5), None, 0.0,
         None, "worst case: random full-range arm poses"),
        ("hands_6kg", S["reach_forward"], Command(lin_vel_x=0.5), None, 6.0, None,
         "6 kg in each hand, arms extended"),
        ("rough_push", S["slam_extremes"], Command(lin_vel_x=0.5), "rough", 0.0,
         [Push(4.0, (-1.5, 0.0, 0.0)), Push(8.0, (0.0, 1.5, 0.0))],
         "rough terrain + 1.5 m/s shoves + arm slams"),
    ]
    for name, sk, cmd, terr, kg, pushes, caption in solo:
        res, frames = run_clip(
            policy, manifest, sk, cmd, masked=True, terrain=terr, hand_kg=kg,
            pushes=pushes, duration=12.0,
            title=name.replace("_", " "), subtitle=caption,
            note="A3 Ultra · FastSAC 50,000 it · arm obs channels frozen · MuJoCo",
        )
        if not res.survived:
            frames += [frames[-1]] * FPS
        write_video(OUT / f"{name}.mp4", frames, fps=FPS)
        print(f"[{'OK  ' if res.survived else 'FALL'}] {name:20s} "
              f"velerr={res.lin_vel_error:.3f} angerr={res.ang_vel_error:.3f} "
              f"tilt={res.max_tilt_deg:.1f}")

    assemble_montages(compare_order, solo_order)
    print(f"\nvideos -> {OUT}")


if __name__ == "__main__":
    main()
