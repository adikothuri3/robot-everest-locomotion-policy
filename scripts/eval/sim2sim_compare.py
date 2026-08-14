"""Side-by-side video: the trained policy vs the PD-stand floor, same disturbance.

Both runs use the same asset, the same harness and the same impulse; the only
difference is the controller. The floor gets the diagnostics' stiff hold gains, so
it is the baseline at its best rather than a straw man.

    python scripts/eval/sim2sim_compare.py --onnx <policy>.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402

from everest_locomotion.robots.manifest import load_manifest  # noqa: E402
from everest_locomotion.evaluation.sim2sim import A3Sim, Command, HolosomaPolicy, Push  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim2sim_suite import StandPolicy, make_font, write_video  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
W, H, FPS = 640, 540, 25
PUSH_T, PUSH_V = 2.5, 1.0


class PanelOverlay:
    def __init__(self, title: str, subtitle: str):
        self.title, self.subtitle = title, subtitle
        self.f_title = make_font(21)
        self.f_sub = make_font(14)
        self.f_data = make_font(16)
        self.frames: list[np.ndarray] = []
        self.fell = False

    def __call__(self, frame, t, cmd, v_body, w_body, tilt, height, alive):
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, img.width, 62], fill=(12, 14, 18, 205))
        d.text((14, 8), self.title, font=self.f_title, fill=(245, 245, 245))
        d.text((14, 36), self.subtitle, font=self.f_sub, fill=(150, 180, 220))

        # The fall thresholds are relaxed here so both panels keep filming to the end,
        # so judge "fallen" from the pose itself rather than from the episode ending.
        if not alive or tilt > 60.0 or height < 0.45:
            self.fell = True
        status = "FALLEN" if self.fell else "UPRIGHT"
        colour = (255, 100, 100) if self.fell else (120, 215, 165)
        lines = [f"t      {t:5.2f} s", f"pelvis {height:.2f} m", f"tilt   {tilt:5.1f}°"]
        d.rectangle([0, img.height - 92, 230, img.height], fill=(12, 14, 18, 205))
        for i, line in enumerate(lines):
            d.text((14, img.height - 84 + 22 * i), line, font=self.f_data, fill=(232, 232, 232))
        d.text((img.width - 110, img.height - 34), status, font=self.f_data, fill=colour)
        self.frames.append(np.asarray(img))


def run_side(policy, manifest, title, subtitle):
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "assets/a3_ultra/holosoma/a3_ultra_29dof.xml")
    )
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    sim = A3Sim(policy, manifest, model=model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.8, 130.0, -10.0
    ov = PanelOverlay(title, subtitle)

    result = sim.run(
        duration_s=8.0,
        command=Command(),
        pushes=[Push(PUSH_T, (PUSH_V, 0.0, 0.0))],
        renderer=renderer,
        camera=cam,
        video_fps=FPS,
        frame_cb=ov,
        # keep filming after the floor topples, so both panels stay in sync
        fall_tilt_deg=179.0,
        fall_height_m=0.05,
    )
    renderer.close()
    return result, ov.frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", default="results/videos/showcase/02_vs_floor.mp4")
    args = p.parse_args()

    manifest = load_manifest()
    policy = HolosomaPolicy(args.onnx)

    _, left = run_side(
        StandPolicy(policy), manifest,
        "PD-stand floor", "stiff hold gains — the recorded baseline",
    )
    _, right = run_side(
        policy, manifest,
        "Learned policy", f"FastSAC, {policy.iteration:,} iterations",
    )

    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))

    from PIL import Image, ImageDraw

    f = make_font(17)
    frames = []
    for i, (a, b) in enumerate(zip(left, right)):
        canvas = Image.new("RGB", (W * 2 + 4, H + 34), (10, 12, 16))
        canvas.paste(Image.fromarray(a), (0, 34))
        canvas.paste(Image.fromarray(b), (W + 4, 34))
        d = ImageDraw.Draw(canvas)
        label = (f"A3 Ultra · identical {PUSH_V:g} m/s impulse at t = {PUSH_T:g} s · "
                 "same robot, same physics, same harness")
        d.text(((canvas.width - d.textlength(label, font=f)) / 2, 8), label, font=f,
               fill=(190, 198, 210))
        frames.append(np.asarray(canvas))

    out = REPO / args.out
    write_video(out, frames, fps=FPS)
    print(f"{out}  ({len(frames)} frames, {len(frames) / FPS:.0f}s)")


if __name__ == "__main__":
    main()
