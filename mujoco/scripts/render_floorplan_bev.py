#!/usr/bin/env python3
"""Render a bird's-eye (BEV) review image of the parametric floorplan.

This is a review/probing tool, not part of the sim loop.  It builds the flat
ring scene from sim.floorplan, adds a temporary top-down camera, renders a
1000x1000 BEV with the ring waypoints/spawn overlaid, and optionally a 3/4
view so wall height is readable.

    cd mujoco && python3 scripts/render_floorplan_bev.py \
        --out ../output/floorplan_bev.png

The container needs an EGL-capable MuJoCo build (mjsim image has one).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent.parent          # .../mujoco
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sim.floorplan import (  # noqa: E402
    FloorPlanParams,
    build_floorplan_spec,
    floorplan_spawns,
    floorplan_waypoints,
    load_params,
)

W, H = 1100, 620
CAM_H = 12.5
FOVY = 22.5


def _project(x: float, y: float) -> tuple[int, int]:
    """World (x, y) -> BEV pixel.  Camera at (0, 0, CAM_H) looking down;
    identity quat, fovy is vertical, square pixels."""
    fx = 0.5 * H / math.tan(math.radians(FOVY) / 2.0)
    u = W / 2.0 + fx * x / CAM_H
    v = H / 2.0 - fx * y / CAM_H
    return int(round(u)), int(round(v))


def _label(img, text, u, v, color=(255, 255, 255), scale=0.62,
           thickness=1, anchor="center", bg=(20, 20, 24), pad=5):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                     scale, thickness)
    if anchor == "center":
        x0, y0 = u - tw // 2, v - th // 2
    elif anchor == "left":
        x0, y0 = u, v - th // 2
    else:                                               # right
        x0, y0 = u - tw, v - th // 2
    cv2.rectangle(img, (x0 - pad, y0 - pad),
                  (x0 + tw + pad, y0 + th + base + pad), bg, -1)
    cv2.putText(img, text, (x0, y0 + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def render_topdown(p: FloorPlanParams, out: Path) -> None:
    spec = build_floorplan_spec(p)
    spec.worldbody.add_camera(name="bev", pos=(0.0, 0.0, CAM_H),
                             quat=(1.0, 0.0, 0.0, 0.0), fovy=FOVY)
    model = spec.compile()
    model.vis.global_.offwidth = max(1280, W)
    model.vis.global_.offheight = max(1280, H)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "bev")

    rend = mujoco.Renderer(model, height=H, width=W)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    rend.update_scene(data, camera=cam_id)
    rgb = np.asarray(rend.render(), dtype=np.uint8)
    img = rgb[:, :, ::-1].copy()                        # RGB -> BGR

    # Four task points (the loop corners).  The ring direction is shown as
    # two offset loops: clockwise = 'down', counterclockwise = 'up'.
    spawns = floorplan_spawns(p)
    targets = floorplan_waypoints(p)
    Lx, Ly = p.free_len_x, p.free_len_y
    Wf, P = p.flight_width, p.platform_len
    cx = Lx / 2.0 - P / 2.0          # W/E platform centreline x
    cy = Ly / 2.0 - Wf / 2.0         # N/S flight centreline y
    delta = 0.18                     # corridor has 0.8 m width; keep clear

    def _ring(d: float, clockwise: bool) -> list[tuple[int, int]]:
        """Corner ring polyline; +d is outward (toward the outer wall)."""
        s = 1.0 if clockwise else -1.0
        xy = [(-cx - d * s, cy + d * s), (cx + d * s, cy + d * s),
              (cx + d * s, -cy - d * s), (-cx - d * s, -cy - d * s),
              (-cx - d * s, cy + d * s)]
        return [_project(x, y) for x, y in xy]

    ov = img.copy()
    for pts, col in ((_ring(+delta, True), (255, 210, 60)),      # cyan = down
                     (_ring(-delta, False), (190, 80, 255))):    # magenta = up
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.arrowedLine(ov, a, b, col, 3, cv2.LINE_AA,
                            tipLength=0.04)
    img = cv2.addWeighted(ov, 0.62, img, 0.38, 0.0)

    # side labels, offset from the target markers
    _label(img, "flight N", *_project(1.0, cy),
           color=(255, 255, 255), scale=0.68, anchor="left")
    _label(img, "flight S", *_project(-1.0, -cy),
           color=(255, 255, 255), scale=0.68, anchor="right")
    _label(img, "platform W", *_project(-cx, 0.78),
           color=(255, 255, 255), scale=0.68)
    _label(img, "platform E", *_project(cx, 0.78),
           color=(255, 255, 255), scale=0.68)
    _label(img, "well (solid)", *_project(0.0, 0.0),
           color=(210, 210, 210), scale=0.66)

    # four task markers at the loop corners
    colors = [(90, 235, 110), (70, 190, 255), (80, 160, 255), (255, 170, 70)]
    for i, (t, col) in enumerate(zip(targets, colors)):
        u, v = _project(*t["pos"][:2])
        cv2.circle(img, (u, v), 13, col, -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 13, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (u - 6, v + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2,
                    cv2.LINE_AA)
        _label(img, t["name"], u + 18, v - 8,
               color=(240, 240, 240), scale=0.5, anchor="left")
    sw = _project(*spawns["w"][0][:2])
    _label(img, "spawn W (yaw +90)", sw[0] - 14, sw[1] + 4,
           color=(120, 255, 140), scale=0.52, anchor="right")

    # go2 footprint at the spawn (0.6 m long x 0.3 m wide, facing +y)
    fp = [(-0.15, -0.30), (0.15, -0.30), (0.15, 0.30), (-0.15, 0.30)]
    fp_px = np.array([_project(-cx + dx, 0.0 + dy) for dx, dy in fp],
                     np.int32)
    cv2.polylines(img, [fp_px], True, (90, 235, 110), 2, cv2.LINE_AA)
    _label(img, "go2 0.6 x 0.3 m", *_project(-cx, -1.08),
           color=(120, 255, 140), scale=0.48)

    # header / footer
    cv2.rectangle(img, (0, 0), (W, 76), (22, 22, 26), -1)
    cv2.putText(img, "Parametric floorplan - BEV (flat ground)",
                (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.76,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img,
                f"flights {p.flight_len:.2f} x {p.flight_width:.2f} m   "
                f"well {p.well_width:.2f} m   "
                f"platforms {p.platform_len:.2f} x {p.ring_width:.2f} m   "
                f"outer {p.outer_len_x:.2f} x {p.outer_len_y:.2f} m   "
                f"targets WN(1)/EN(2)/ES(3)/WS(4)",
                (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (190, 210, 220), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, H - 55), (W, H), (22, 22, 26), -1)
    cv2.putText(img,
                "cyan CW = down (WN->EN->ES->WS)   "
                "magenta CCW = up (WN->WS->ES->EN)   all four = task points",
                (18, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (210, 220, 225), 1, cv2.LINE_AA)

    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise SystemExit(f"failed to write {out}")
    rend.close()
    print(f"BEV -> {out}  ({W}x{H})  {p.summary()}", flush=True)


def render_three_quarter(p: FloorPlanParams, out: Path) -> None:
    spec = build_floorplan_spec(p)
    model = spec.compile()
    model.vis.global_.offwidth = max(1280, W)
    model.vis.global_.offheight = max(1280, H)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.0, 0.0, 0.35])
    cam.distance = 8.5
    cam.azimuth = 135.0
    cam.elevation = -32.0
    rend = mujoco.Renderer(model, height=H, width=W)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    rend.update_scene(data, camera=cam)
    rgb = np.asarray(rend.render(), dtype=np.uint8)
    img = rgb[:, :, ::-1].copy()
    cv2.rectangle(img, (0, 0), (W, 76), (22, 22, 26), -1)
    cv2.putText(img, "Parametric floorplan - 3/4 view",
                (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.76,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img,
                f"outer {p.outer_len_x:.2f} x {p.outer_len_y:.2f} m, "
                f"wall {p.wall_height:.2f} m, flat floor (z=0)",
                (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (190, 210, 220), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise SystemExit(f"failed to write {out}")
    rend.close()
    print(f"3/4 -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="floorplan YAML (default: assets/floorplan/default.yaml)")
    ap.add_argument("--out", default="floorplan_bev.png")
    ap.add_argument("--out3q", default=None,
                    help="optional 3/4 review image path")
    args = ap.parse_args()

    p = load_params(args.config)
    render_topdown(p, Path(args.out))
    if args.out3q:
        render_three_quarter(p, Path(args.out3q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
