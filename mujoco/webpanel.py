#!/usr/bin/env python3
"""Operator panel on the viser web viewer, shared by the hosts.

tinynav-sim's own controls (the vendored mjviser under webviewer/ stays
untouched), drawn above its Scene / Visualization / Groups tabs:

  Robot           registry robots that have a body and a policy
                  (sim/robots.py); the others are listed with what they miss
  Perception      the robot's depth policy (default) or its blind one; locked
                  while the robot has only one of the two
  Gamepad         sim/gamepad.py control on/off (A on the pad toggles too)
  Follow camera   chase view straight behind and above the robot (default
                  on); off hands the camera back to the mouse
  Reset / Stop    the native viewer's R / Space
  policy depth    the policy's newest 60x86 depth frame (perception only),
                  turbo map: near red, DEPTH_MAX_M (3 m) and beyond dark blue

The panel sits in viser's control panel, i.e. the top-right corner of the
page. Widget callbacks run on viser's thread, so robot and perception
changes are only queued (`take_request()`); the host applies them on its
control thread between ticks and reports back with `show_robot()`.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

FOLLOW_DISTANCE = 2.6     # m from the base
FOLLOW_PITCH_DEG = 40.0   # looking down
FOLLOW_TAU_POS = 0.15     # s, position low-pass
FOLLOW_TAU_YAW = 0.6      # s, heading low-pass (no gait wobble)
DEPTH_HZ = 10.0           # = the policy's depth rate
DEPTH_UPSCALE = 3         # 60x86 -> 180x258, nearest
STATUS_HZ = 2.0


def base_yaw(quat_wxyz) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def depth_rgb(frame: np.ndarray) -> np.ndarray:
    """Policy depth (depth / DEPTH_MAX_M, 0..1) -> RGB, near = red."""
    import cv2
    d = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
    bgr = cv2.applyColorMap(((1.0 - d) * 255.0).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return np.repeat(np.repeat(rgb, DEPTH_UPSCALE, 0), DEPTH_UPSCALE, 1)


class FollowCamera:
    """Chase pose straight behind and above the base, low-passed."""

    def __init__(self, distance: float = FOLLOW_DISTANCE,
                 pitch_deg: float = FOLLOW_PITCH_DEG) -> None:
        self.distance = distance
        self.pitch = math.radians(pitch_deg)
        self._pos: np.ndarray | None = None
        self._yaw = 0.0

    def reset(self) -> None:
        self._pos = None

    def pose(self, base_pos, yaw: float, dt: float):
        base_pos = np.asarray(base_pos, dtype=np.float64)
        if self._pos is None:
            self._pos, self._yaw = base_pos.copy(), yaw
        else:
            a = 1.0 - math.exp(-dt / FOLLOW_TAU_POS)
            b = 1.0 - math.exp(-dt / FOLLOW_TAU_YAW)
            self._pos += a * (base_pos - self._pos)
            self._yaw += b * math.remainder(yaw - self._yaw, math.tau)
        back = self.distance * math.cos(self.pitch)
        offset = np.array((-back * math.cos(self._yaw), -back * math.sin(self._yaw),
                           self.distance * math.sin(self.pitch)))
        return self._pos + offset, self._pos.copy()


class OperatorPanel:
    def __init__(self, server, robots, robot: str, *, gamepad=None,
                 on_button=None, robot_select: bool = True) -> None:
        """robots: {name: sim.robots.Robot}; robot: the one being driven.
        on_button("reset" | "stop") is called from viser's thread.
        robot_select=False shows the robot without the dropdown (hosts that
        cannot swap bodies at runtime)."""
        self.server = server
        self.robots = robots
        self.gamepad = gamepad
        self.follow_cam = FollowCamera()
        self._lock = threading.Lock()
        self._request: tuple[str, str | bool] | None = None
        self._quiet = False        # programmatic widget update in progress
        self._t_depth = self._t_status = self._t_cam = 0.0
        self._last_pose = None     # (camera, look_at) of the last update
        self._robot = robot

        selectable = [n for n, r in robots.items() if not r.missing()]
        missing = {n: r.missing() for n, r in robots.items() if r.missing()}
        self._by_label = {robots[n].label: n for n in selectable}
        gui = server.gui
        # order -1: above mjviser's tabs whichever was built first
        with gui.add_folder("Sim", order=-1.0):
            if robot_select:
                self.robot_dd = gui.add_dropdown(
                    "Robot", [robots[n].label for n in selectable],
                    initial_value=robots[robot].label,
                    hint="Switching reloads the body and its policy.")
                self.robot_dd.on_update(self._on_robot)
                if missing:
                    gui.add_markdown("<small>Not available: " + "; ".join(
                        f"{robots[n].label} ({', '.join(m)})"
                        for n, m in missing.items()) + "</small>")
            else:
                self.robot_dd = None
                gui.add_markdown(f"Robot: **{robots[robot].label}** "
                                 "<small>(chosen at boot: --robot)</small>")
            self.perception = gui.add_checkbox("Perception", True)
            self.perception.on_update(self._on_perception)
            self.pad_cb = gui.add_checkbox(
                "Gamepad", bool(gamepad and gamepad.enabled),
                disabled=gamepad is None,
                hint="Left stick vx/vy, right stick wz; A on the pad toggles, "
                     "B stops. Read from the sim host's /dev/input.")
            self.pad_cb.on_update(self._on_gamepad)
            self.follow = gui.add_checkbox(
                "Follow camera", True,
                hint="Chase view behind and above the robot; off frees the camera.")
            self.follow.on_update(lambda _e: self.follow_cam.reset())
            for label, name in (("Reset to spawn", "reset"), ("Stop", "stop")):
                btn = gui.add_button(label)
                btn.on_click(lambda _e, _n=name: on_button and on_button(_n))
            self.depth = gui.add_image(
                np.zeros((60 * DEPTH_UPSCALE, 86 * DEPTH_UPSCALE, 3), np.uint8),
                label="Policy depth", format="jpeg", jpeg_quality=85)
            self.depth_note = gui.add_markdown(
                "<small>policy depth input, 60x86 at 10 Hz: near red, 3 m+ blue</small>")
            self.status = gui.add_markdown("")
        server.on_client_connect(self._frame_new_client)

    # -- viser-thread callbacks ------------------------------------------------
    def _on_robot(self, _event) -> None:
        if self._quiet:
            return
        name = self._by_label.get(self.robot_dd.value)
        if name is not None and name != self._robot:
            with self._lock:
                self._request = ("robot", name)

    def _on_perception(self, _event) -> None:
        if not self._quiet:
            with self._lock:
                self._request = ("perception", bool(self.perception.value))

    def _on_gamepad(self, _event) -> None:
        if self.gamepad is not None and not self._quiet:
            self.gamepad.enabled = bool(self.pad_cb.value)

    def _frame_new_client(self, client) -> None:
        # Follow on: the next update() frames it. Off: start at the last
        # chase pose once instead of the far-away mjviser default.
        pose = self._last_pose
        if not self.follow.value and pose is not None:
            client.camera.position, client.camera.look_at = pose

    # -- host (control thread) -------------------------------------------------
    def take_request(self):
        """("robot", name) | ("perception", on) | None; the newest wins."""
        with self._lock:
            request, self._request = self._request, None
        return request

    def show_robot(self, robot: str, perception: bool, has_perception: bool,
                   has_blind: bool) -> None:
        """Reflect the driven robot/policy in the widgets (after a switch or a
        refused one)."""
        self._robot = robot
        self._quiet = True
        try:
            if self.robot_dd is not None:
                self.robot_dd.value = self.robots[robot].label
            self.perception.value = perception
            self.perception.disabled = not (has_perception and has_blind)
            self.perception.hint = (
                None if has_perception and has_blind else
                f"{self.robots[robot].label} has no "
                f"{'blind' if has_perception else 'depth'} policy yet")
            self.depth.visible = self.depth_note.visible = perception
        finally:
            self._quiet = False
        self.follow_cam.reset()

    def update(self, base_pos, quat_wxyz, depth_frame, status: str) -> None:
        """Per render tick: chase camera, depth inset (10 Hz), status line."""
        now = time.monotonic()
        dt = min(now - self._t_cam, 0.5) if self._t_cam else 0.0
        self._t_cam = now
        cam, look = self.follow_cam.pose(base_pos, base_yaw(quat_wxyz), dt)
        self._last_pose = (cam, look)
        if self.follow.value:
            for client in self.server.get_clients().values():
                client.camera.position = cam
                client.camera.look_at = look
        if depth_frame is not None and now - self._t_depth >= 1.0 / DEPTH_HZ:
            self._t_depth = now
            self.depth.image = depth_rgb(depth_frame)
        if self.gamepad is not None and self.pad_cb.value != self.gamepad.enabled:
            self._quiet = True
            try:
                self.pad_cb.value = self.gamepad.enabled   # A pressed on the pad
            finally:
                self._quiet = False
        if now - self._t_status >= 1.0 / STATUS_HZ:
            self._t_status = now
            self.status.content = f"<small>{status}</small>"


__all__ = ["FollowCamera", "OperatorPanel", "base_yaw", "depth_rgb"]
