#!/usr/bin/env python3
"""Operator panel on the viser web viewer, shared by the hosts.

tinynav-sim's own controls (the vendored mjviser under webviewer/ stays
untouched), drawn above its Scene / Visualization / Groups tabs:

  Map             the scene registry (sim/plant.py SCENES: map3 楼梯井, map2
                  室内, flat 平地); a switch rebuilds the robot in the new map at
                  its spawn and frames every page behind it again
  Robot           registry robots that have a body and a policy
                  (sim/robots.py); the others are listed with what they miss
  Perception      the robot's depth policy (default) or its blind one; locked
                  while the robot has only one of the two
  Gamepad         sim/gamepad.py control on/off (A on the pad toggles too)
  Track robot     the view follows the robot (default on): the scene is drawn
                  shifted so the robot stays at the viewer origin and turned
                  against its heading, so the view keeps its angle to the
                  robot as it turns; the page's own controls orbit it (left
                  drag), zoom (wheel) and pan (right drag); a camera that
                  would see the robot through a floor or a flight is pulled
                  in front of it, and eases back out once clear. Off: the
                  scene stops moving and the camera stays put, to move with
                  the mouse or W/A/S/D/Q (the vendored client's camera keys)
  Wall guard      no forward speed into a wall the policy's depth camera sees
                  (sim/guard.py, default on): slows down and stands ~0.5 m in
                  front of it; the status line says when it cuts the command
  Keyboard speed  held teleop keys drive this share of the active policy's
                  training command range (sim/keyboard.py, default 50%)
  Max accel       the keyboard's and the gamepad's command changes at most
  Max yaw accel   this fast: m/s² for vx and vy together, rad/s² for wz,
                  speeding up and braking alike; the pad's B still stops at
                  once (sim/accel.py, default 1.0 / 2.0)
  controls        one line: the teleop keys with their current speeds
  Reset / Stop    the native viewer's R / Space
  policy depth    the policy's newest depth frame (perception only; PIE's
                  60x86, a navcore student's 80x68 portrait), turbo map:
                  near red, its max range (3 m) and beyond dark blue, like
                  what navcore cannot see (nearer than 0.15 m)
  Hip temperature the four hip motors' estimated winding temperatures
                  (sim/thermal.py: from the simulated joint torques,
                  parameters not identified on the robot), live over the last
                  TEMP_WINDOW_S like a monitor trace

The depth policy's camera is drawn on the robot as a small pyramid while
the perception policy drives (apex at the lens, opening = what the policy
sees; add_depth_camera): the hosts show it by the policy kind.

The panel sits in viser's control panel, i.e. the top-right corner of the
page. Widget callbacks run on viser's thread, so robot and perception
changes are only queued (`take_request()`); the host applies them on its
control thread between ticks and reports back with `show_robot()`.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

FRAME_DISTANCE = 1.5      # m, a new page's camera from the base ...
FRAME_LENGTHS = 1.8       # ... or this many robot lengths, given one
FRAME_PITCH_DEG = 30.0    # behind the robot, looking down
TRACK_TAU = 0.15          # s, low-pass on the followed base and heading (gait
                          # wobble: Go2 1.3 deg -> 0.8; 6 deg behind a 46 deg/s turn)
TRACK_SNAP_M = 1.0        # a jump this long (reset, a fall) is not eased ...
TRACK_SNAP_YAW = 1.0      # ... nor a turn of this many rad between two updates
SIGHT_MARGIN = 0.15       # m, camera kept in front of what would hide the robot
SIGHT_MIN = 0.3           # m, nearest the camera is pulled to it
SIGHT_TAU_OUT = 0.3       # s, easing back out once nothing hides it
DEPTH_HZ = 10.0           # = the policy's depth rate
DEPTH_UPSCALE = 3         # 60x86 -> 180x258 (80x68 -> 240x204), nearest
STATUS_HZ = 10.0          # status line (cmd, upright, gamepad)
DEPTH_CAMERA_LENGTHS = 0.2   # the depth camera's pyramid: this many robot
DEPTH_CAMERA_DEPTH = 0.15    # lengths deep, or this many m without one
DEPTH_CAMERA_RGB = (40, 160, 255)
TEMP_HZ = 10.0             # hip temperature trace: points per second ...
TEMP_WINDOW_S = 20.0       # ... over a fixed window this long (s)
TEMP_COLORS = {"FL": "#e6550d", "FR": "#3182bd", "RL": "#fdae6b", "RR": "#9ecae1"}


def base_yaw(quat_wxyz) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def z_rotation(yaw: float) -> np.ndarray:
    """3x3 rotation by `yaw` about +z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def add_depth_camera(scene, length: float | None = None, depth_spec=None) -> None:
    """Draw the policy's depth camera on `scene` (a webviewer
    ViserMujocoScene of a sim.plant model) as a small pyramid riding the
    base: apex at the lens, opening = the camera's fovy by the policy's frame
    (depth_spec: a sim.contract_v2.DepthSpec; PIE's crop of the render,
    DEPTH_WIDTH x DEPTH_HEIGHT, without one). length: the robot's (m)."""
    from sim import contract as C
    from sim.plant import DEPTH_CAMERA_NAME
    aspect = (depth_spec.width / depth_spec.height if depth_spec is not None
              else C.DEPTH_WIDTH / C.DEPTH_HEIGHT)
    scene.add_camera_frustum(
        DEPTH_CAMERA_NAME, aspect=aspect,
        depth=DEPTH_CAMERA_LENGTHS * length if length else DEPTH_CAMERA_DEPTH,
        color=DEPTH_CAMERA_RGB, variant="filled",
        thickness=2.0, thickness_units="screen")


def sync_depth_camera(scene, model, depth_spec, length: float | None = None) -> None:
    """The policy's depth camera moved (a navcore student's mount and field of
    view set on the plant `model` by its runtime): copy it into the web view's
    model and redraw the pyramid."""
    import mujoco

    from sim.plant import DEPTH_CAMERA_NAME
    src = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAMERA_NAME)
    dst = mujoco.mj_name2id(scene.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAMERA_NAME)
    if src < 0 or dst < 0:
        return
    scene.mj_model.cam_pos[dst] = model.cam_pos[src]
    scene.mj_model.cam_quat[dst] = model.cam_quat[src]
    scene.mj_model.cam_fovy[dst] = model.cam_fovy[src]
    scene.clear_camera_frustums()
    add_depth_camera(scene, length, depth_spec)


def march_status(march: bool, contract) -> str:
    """Status-line suffix while marching in place is on (policies trained
    for it: contract.gait_march)."""
    if not march:
        return ""
    return " · 踏步" if getattr(contract, "gait_march", False) else " · 踏步（此策略不支持）"


def guard_status(guard) -> str:
    """Status-line suffix while the wall guard (sim/guard.py) cuts the
    forward command."""
    if guard is None or not guard.limiting:
        return ""
    return f" · 前方墙 {guard.distance:.2f} m，限速"


def depth_note(shape) -> str:
    """The line under the depth inset: the frame's size (rows x columns)."""
    return f"<small>{shape[0]}×{shape[1]} · 10 Hz · 近红远蓝</small>"


def depth_rgb(frame: np.ndarray) -> np.ndarray:
    """Policy depth (depth / max range, 0..1) -> RGB, near = red."""
    import cv2
    d = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
    bgr = cv2.applyColorMap(((1.0 - d) * 255.0).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return np.repeat(np.repeat(rgb, DEPTH_UPSCALE, 0), DEPTH_UPSCALE, 1)


class Tracker:
    """The followed pose: the base and its heading (the yaw of its x axis),
    low-passed; a jump of TRACK_SNAP_M or more (reset, a fall) is taken at
    once, and so is a turn of TRACK_SNAP_YAW between two updates. The
    heading is unwrapped (runs on past +-pi) and held while the base points
    steeply up or down, where it is not defined."""

    def __init__(self) -> None:
        self.pos: np.ndarray | None = None
        self.yaw: float | None = None

    def update(self, base_pos, quat_wxyz, dt: float) -> tuple[np.ndarray, float]:
        base_pos = np.asarray(base_pos, dtype=np.float64)
        jump = self.pos is None or np.linalg.norm(base_pos - self.pos) >= TRACK_SNAP_M
        ease = 1.0 - math.exp(-dt / TRACK_TAU)
        if jump:
            self.pos = base_pos.copy()
        else:
            self.pos = self.pos + ease * (base_pos - self.pos)
        w, x, y, z = (float(v) for v in quat_wxyz)
        level = math.hypot(1.0 - 2.0 * (y * y + z * z), 2.0 * (w * z + x * y))
        if self.yaw is None:
            self.yaw = base_yaw(quat_wxyz)
        elif level >= 0.5:                       # x axis within 60 deg of level
            turn = math.remainder(base_yaw(quat_wxyz) - self.yaw, math.tau)
            self.yaw += turn if jump or abs(turn) >= TRACK_SNAP_YAW else ease * turn
        return self.pos, self.yaw


class SightKeeper:
    """Keeps one page's camera in sight of its orbit target (the robot).

    The page owns the camera; `pref` is the orbit offset (camera minus
    target) its user chose, read back from the page's camera reports. While
    the line from the target to the camera crosses a surface the view cannot
    see through, the camera is held in front of it on that line -- pulled in
    at once, eased back out to `pref` once clear. A report off the held line
    is the user turning (new direction) or zooming in. Placements made here
    come back as reports too (late, or with a half-applied target) and are
    recognised: the page ends up at the newest one."""

    def __init__(self) -> None:
        self.pref: np.ndarray | None = None
        self.arm: float | None = None          # distance held along pref
        self._placed: list[tuple[np.ndarray, np.ndarray]] = []  # (camera, target)

    def place(self, camera, position, look_at) -> None:
        position = np.array(position, dtype=np.float64)
        look_at = np.array(look_at, dtype=np.float64)
        camera.position = position   # viser moves look_at along with it ...
        camera.look_at = look_at     # ... so aim it back
        self._placed = self._placed[-63:] + [(position, look_at)]

    def frame(self, camera, position, look_at) -> None:
        """Place the camera and take that as the user's choice."""
        self.place(camera, position, look_at)
        self.pref = self._placed[-1][0] - self._placed[-1][1]
        self.arm = float(np.linalg.norm(self.pref))

    def step(self, camera, offset, yaw: float, ray, dt: float) -> None:
        """offset, yaw: the scene transform, viewer = Rz(yaw) (world +
        offset); ray(origin, unit direction, reach) -> distance to the first
        blocking surface (world frame) or None."""
        pos = np.array(camera.position, dtype=np.float64)
        look = np.array(camera.look_at, dtype=np.float64)
        mine = any(np.allclose(pos, p, atol=1e-3) for p, _ in self._placed)
        if mine:   # an echo: the page ends up at the newest placement
            pos, look = self._placed[-1]
        orbit = pos - look
        dist = float(np.linalg.norm(orbit))
        if dist < 1e-6:
            return
        if not mine or self.pref is None:
            want = float(np.linalg.norm(self.pref)) if self.pref is not None else 0.0
            if self.arm is not None and self.arm < want - 1e-3 and dist > self.arm - 0.02:
                # turned (or zoomed out, which a held camera cannot) while
                # held in: the new direction at the preferred distance
                self.pref = orbit * (want / dist)
            else:
                self.pref, self.arm = orbit, dist    # placed (or zoomed in) by the user
        want = float(np.linalg.norm(self.pref))
        direction = self.pref / want
        back = z_rotation(-yaw)                  # viewer -> world
        hit = ray(back @ look - offset, back @ direction, want + SIGHT_MARGIN)
        allowed = want if hit is None else max(min(want, hit - SIGHT_MARGIN), SIGHT_MIN)
        if allowed < self.arm:
            arm = allowed                            # in at once: never see through
        else:
            arm = self.arm + (allowed - self.arm) * (1.0 - math.exp(-dt / SIGHT_TAU_OUT))
            if allowed - arm < 0.005:
                arm = allowed
        self.arm = arm
        # in: right away; out: in 2 cm steps, then the last bit
        if dist - arm > 1e-3 or arm - dist > 0.02 or (arm == allowed and arm - dist > 1e-3):
            self.place(camera, look + direction * arm, look)


class OperatorPanel:
    def __init__(self, server, robots, robot: str, *, gamepad=None,
                 keyboard=None, accel=None, guard=None, on_button=None,
                 robot_select: bool = True, scenes: dict[str, str] | None = None,
                 scene: str | None = None) -> None:
        """robots: {name: sim.robots.Robot}; robot: the one being driven.
        keyboard: the host's sim.keyboard.HoldCmd (its envelope already set).
        accel: the host's sim.accel.AccelLimit over its teleop commands;
        guard: its sim.guard.WallGuard.
        on_button("reset" | "stop") is called from viser's thread.
        robot_select=False shows the robot without the dropdown (hosts that
        cannot swap bodies at runtime). scenes: {name: label} of the maps a
        switch can load, in the dropdown's order; scene: the one loaded
        (without scenes it is shown as chosen at boot)."""
        self.server = server
        self.robots = robots
        self.gamepad = gamepad
        self.keyboard = keyboard
        self.accel = accel
        self.guard = guard
        # viewer = Rz(scene_yaw) (world + scene_offset); hosts draw the
        # scene with it
        self.scene_offset = np.zeros(3)
        self.scene_yaw = 0.0
        self._view_heading: float | None = None   # the robot's, in the viewer, while tracking
        self._tracker = Tracker()
        self._keepers: dict[int, SightKeeper] = {}   # per client
        self._frame_distance = FRAME_DISTANCE
        self._lock = threading.Lock()
        self._request: tuple[str, str | bool] | None = None
        self._loading: list = []   # "switching robot" spinners, one per client
        self._to_frame: set[int] = set()   # new pages, framed on the next update
        self._aim = False          # Track robot just turned on
        self._quiet = False        # programmatic widget update in progress
        self._t_depth = self._t_status = self._t_cam = 0.0
        self._robot = robot
        self._scene = scene

        selectable = [n for n, r in robots.items() if not r.missing()]
        missing = {n: r.missing() for n, r in robots.items() if r.missing()}
        self._by_label = {robots[n].label: n for n in selectable}
        gui = server.gui
        # order -1: above mjviser's tabs whichever was built first
        with gui.add_folder("Sim", order=-1.0):
            self._scene_by_label = {label: name for name, label in (scenes or {}).items()}
            self.scene_dd = None
            if scenes and len(scenes) > 1:
                self.scene_dd = gui.add_dropdown(
                    "Map", list(scenes.values()),
                    initial_value=scenes.get(scene, next(iter(scenes.values()))),
                    hint="Switching loads the map and stands the robot at its spawn.")
                self.scene_dd.on_update(self._on_scene)
            elif scene is not None:
                gui.add_markdown(f"Map: **{(scenes or {}).get(scene, scene)}** "
                                 "<small>(chosen at boot: --scene)</small>")
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
                     "B stops, X / Y marches in place. Read from the sim "
                     "host's /dev/input.")
            self.pad_cb.on_update(self._on_gamepad)
            self.follow = gui.add_checkbox(
                "Track robot", True,
                hint="The view follows the robot and turns with it: left "
                     "drag orbits it, the wheel zooms, right drag pans. Off: "
                     "the camera stays put; move it with the mouse or "
                     "W/A/S/D/Q.")
            self.follow.on_update(self._on_follow)
            if guard is not None:
                self.guard_cb = gui.add_checkbox(
                    "Wall guard", guard.enabled,
                    hint="No forward speed into a wall the depth camera sees: "
                         "the robot slows down and stands about 0.5 m in front "
                         "of it (the depth students try to climb walls). "
                         "Backing off and turning stay free.")
                self.guard_cb.on_update(self._on_guard)
            for label, name in (("Reset to spawn", "reset"), ("Stop", "stop")):
                btn = gui.add_button(label)
                btn.on_click(lambda _e, _n=name: on_button and on_button(_n))
            if keyboard is not None:
                self.kb_speed = gui.add_slider(
                    "Keyboard speed", min=10, max=100, step=5,
                    initial_value=round(100 * keyboard.speed),
                    hint="Held teleop keys drive this % of the active "
                         "policy's training command range.")
                self.kb_speed.on_update(self._on_kb_speed)
            if accel is not None:
                self.accel_lin = gui.add_slider(
                    "Max accel", min=0.2, max=3.0, step=0.1,
                    initial_value=accel.linear,
                    hint="m/s²: the keyboard's and the gamepad's vx/vy command "
                         "changes at most this fast, speeding up and braking "
                         "(the pad's B still stops at once).")
                self.accel_ang = gui.add_slider(
                    "Max yaw accel", min=0.5, max=6.0, step=0.5,
                    initial_value=accel.angular,
                    hint="rad/s²: the same for the turn command wz.")
                self.accel_lin.on_update(self._on_accel)
                self.accel_ang.on_update(self._on_accel)
            # sim/keyboard.py's hold-to-move keys (read from the host's
            # /dev/input) and the vendored client's camera keys.
            self.keys = gui.add_markdown(self._keys_text())
            self.status = gui.add_markdown("")
            self.depth = gui.add_image(
                np.zeros((60 * DEPTH_UPSCALE, 86 * DEPTH_UPSCALE, 3), np.uint8),
                label="Policy depth", format="jpeg", jpeg_quality=85)
            self._depth_shape = (60, 86)
            self.depth_note = gui.add_markdown(depth_note(self._depth_shape))
        from viser import uplot
        self._temp: deque[tuple[float, np.ndarray]] = deque()
        self._t_temp = -math.inf
        with gui.add_folder("Hip temperature (est.)", order=-0.9):
            self.temp_now = gui.add_markdown("")
            # x = seconds before now, over a fixed window: a monitor trace
            self.temp_plot = gui.add_uplot(
                (np.zeros(1), *(np.zeros(1) for _ in TEMP_COLORS)),
                (uplot.Series(label="s"),
                 *(uplot.Series(label=leg, stroke=color, width=1.5)
                   for leg, color in TEMP_COLORS.items())),
                scales={"x": uplot.Scale(time=False, auto=False, range=(-TEMP_WINDOW_S, 0.0)),
                        "y": uplot.Scale(auto=True)},
                axes=(uplot.Axis(), uplot.Axis(label="°C")),
                legend=uplot.Legend(show=False),
                aspect=1.6)
        server.on_client_connect(self._frame_new_client)

    # -- viser-thread callbacks ------------------------------------------------
    def _on_scene(self, _event) -> None:
        if self._quiet:
            return
        name = self._scene_by_label.get(self.scene_dd.value)
        if name is not None and name != self._scene:
            with self._lock:
                self._request = ("scene", name)
                self._show_loading(f"正在切换到 {self.scene_dd.value}", "加载地图和机身…")

    def _on_robot(self, _event) -> None:
        if self._quiet:
            return
        name = self._by_label.get(self.robot_dd.value)
        if name is not None and name != self._robot:
            with self._lock:
                self._request = ("robot", name)
                self._show_loading(f"正在切换到 {self.robot_dd.value}", "加载机身和策略…")

    def _show_loading(self, title: str, body: str) -> None:
        """A spinner on every page right away, from this (viser) thread: the
        control thread stands still while it loads (a robot ~0.3 s, a map a
        few s). Under self._lock."""
        for handle in self._loading:
            handle.remove()
        self._loading = [
            client.add_notification(title=title, body=body, loading=True,
                                    with_close_button=False)
            for client in self.server.get_clients().values()]

    def _on_perception(self, _event) -> None:
        if not self._quiet:
            with self._lock:
                self._request = ("perception", bool(self.perception.value))

    def _on_gamepad(self, _event) -> None:
        if self.gamepad is not None and not self._quiet:
            self.gamepad.enabled = bool(self.pad_cb.value)

    def _on_follow(self, _event) -> None:
        if self.follow.value:
            with self._lock:
                self._aim = True

    def _on_kb_speed(self, _event) -> None:
        self.keyboard.set_speed(self.kb_speed.value / 100.0)
        self.keys.content = self._keys_text()

    def _on_accel(self, _event) -> None:
        self.accel.set_limits(self.accel_lin.value, self.accel_ang.value)

    def _on_guard(self, _event) -> None:
        self.guard.enabled = bool(self.guard_cb.value)

    def _keys_text(self) -> str:
        if self.keyboard is not None:
            fwd, back, vy, wz = self.keyboard.hold_speeds()
        else:
            from sim.keyboard import VX_HOLD, VX_HOLD_BWD, VY_HOLD, WZ_HOLD
            fwd, back, vy, wz = VX_HOLD, VX_HOLD_BWD, VY_HOLD, WZ_HOLD
        return (f"<small>键盘 ↑/↓ {fwd:.2f}/{back:.2f} · ←/→ {wz:.2f} · "
                f"<b>,</b>/<b>.</b> {vy:.2f} · M 踏步 · 手柄 X 踏步</small>")

    def _frame_new_client(self, client) -> None:
        # framed by the next update(), on the control thread, behind the robot
        with self._lock:
            self._to_frame.add(client.client_id)

    # -- host (control thread) -------------------------------------------------
    def take_request(self):
        """("scene", name) | ("robot", name) | ("perception", on) | None; the
        newest wins."""
        with self._lock:
            request, self._request = self._request, None
        return request

    def show_robot(self, robot: str, perception: bool, has_perception: bool,
                   has_blind: bool, length: float | None = None,
                   scene: str | None = None) -> None:
        """Reflect the driven robot/policy (and the loaded map, scene) in the
        widgets (after a switch or a refused one). length: the robot's
        nose-to-tail span (m), which sets how far a new page's camera starts."""
        self._robot = robot
        self._frame_distance = (FRAME_LENGTHS * length if length
                                else FRAME_DISTANCE)
        self._quiet = True
        try:
            if scene is not None:
                self._scene = scene
                if self.scene_dd is not None:
                    label = next((lb for lb, n in self._scene_by_label.items() if n == scene), None)
                    if label is not None:
                        self.scene_dd.value = label
            if self.robot_dd is not None:
                self.robot_dd.value = self.robots[robot].label
            self.perception.value = perception
            self.perception.disabled = not (has_perception and has_blind)
            self.perception.hint = (
                None if has_perception and has_blind else
                f"{self.robots[robot].label} has no "
                f"{'blind' if has_perception else 'depth'} policy yet")
            self.depth.visible = self.depth_note.visible = perception
            self.keys.content = self._keys_text()   # the new policy's range
        finally:
            self._quiet = False
        # open pages keep their camera: a switch keeps the robot's place
        with self._lock:
            if self._request is not None:      # another switch queued: keep it
                return
            loading, self._loading = self._loading, []
        for handle in loading:
            handle.remove()

    def update(self, base_pos, quat_wxyz, depth_frame, status: str,
               ray=None) -> tuple[np.ndarray, float]:
        """Per render tick: follow the robot, depth inset (10 Hz), status
        line. Returns the scene transform to draw with, (offset, yaw):
        viewer = Rz(yaw) (world + offset). ray(origin, unit direction,
        reach) -> the distance to the first surface the view cannot see
        through, or None (world frame); without it a camera may end up
        behind a floor."""
        now = time.monotonic()
        dt = min(now - self._t_cam, 0.5) if self._t_cam else 0.0
        self._t_cam = now
        pivot, heading = self._tracker.update(base_pos, quat_wxyz, dt)
        tracking = bool(self.follow.value)
        with self._lock:
            new, self._to_frame = self._to_frame, set()
            aim, self._aim = self._aim, False
        if tracking:
            if aim or self._view_heading is None:
                # turned on: from here the view turns with the robot, at
                # the angle it has to it now
                self._view_heading = self.scene_yaw + heading
            self.scene_offset = -pivot
            self.scene_yaw = self._view_heading - heading
        self._follow_cameras(
            new, aim, z_rotation(self.scene_yaw) @ (pivot + self.scene_offset),
            base_yaw(quat_wxyz) + self.scene_yaw, tracking, ray, dt)
        if depth_frame is not None and now - self._t_depth >= 1.0 / DEPTH_HZ:
            self._t_depth = now
            self.depth.image = depth_rgb(depth_frame)
            shape = tuple(np.shape(depth_frame)[-2:])
            if shape != self._depth_shape:      # another camera's policy
                self._depth_shape = shape
                self.depth_note.content = depth_note(shape)
        if self.gamepad is not None and self.pad_cb.value != self.gamepad.enabled:
            self._quiet = True
            try:
                self.pad_cb.value = self.gamepad.enabled   # A pressed on the pad
            finally:
                self._quiet = False
        if now - self._t_status >= 1.0 / STATUS_HZ:
            self._t_status = now
            self.status.content = f"<small>{status}</small>"
        return self.scene_offset, self.scene_yaw

    def reframe(self) -> None:
        """Frame every open page behind the robot again on the next update,
        as a new page (a map switch: the robot is somewhere else now)."""
        with self._lock:
            self._to_frame.update(self.server.get_clients())
            self._aim = True

    def reset_temperatures(self) -> None:
        """A new robot: the hip temperature trace starts over."""
        self._temp.clear()
        self._t_temp = -math.inf

    def plot_temperatures(self, t: float, labels, temps) -> None:
        """Per control tick: t (s, sim time), the hips' labels (FL, FR, RL,
        RR) and temperatures (deg C); drawn at TEMP_HZ over the last
        TEMP_WINDOW_S, x = seconds before now."""
        if t - self._t_temp < 1.0 / TEMP_HZ - 1e-9:
            return
        self._t_temp = t
        values = dict(zip(labels, np.asarray(temps, dtype=np.float64)))
        self._temp.append((float(t), np.asarray([values.get(leg, np.nan) for leg in TEMP_COLORS])))
        while t - self._temp[0][0] > TEMP_WINDOW_S:
            self._temp.popleft()
        times = np.asarray([s for s, _ in self._temp]) - t
        self.temp_plot.data = (times, *np.stack([v for _, v in self._temp], axis=1))
        # the legend: each leg's number in its trace's colour
        self.temp_now.content = "<small>" + " · ".join(
            f'<b style="color:{color}">{leg}</b> {values[leg]:.1f}'
            for leg, color in TEMP_COLORS.items() if leg in values) + " °C</small>"

    def _follow_cameras(self, new: set, aim: bool, robot, yaw: float,
                        tracking: bool, ray, dt: float) -> None:
        """new: the pages to frame; aim: Track robot just turned on. robot,
        yaw: the followed point and the robot's heading in the viewer frame.
        New pages start behind it; Track robot turned on re-centres every
        page on it, keeping each one's orbit; while tracking, pages are kept
        in sight of it."""
        clients = self.server.get_clients()
        if aim:
            self._keepers = {}
        for cid in [c for c in self._keepers if c not in clients]:
            del self._keepers[cid]
        pitch = math.radians(FRAME_PITCH_DEG)
        behind = self._frame_distance * np.array(
            (-math.cos(pitch) * math.cos(yaw), -math.cos(pitch) * math.sin(yaw),
             math.sin(pitch)))
        for cid, client in clients.items():
            keeper = self._keepers.setdefault(cid, SightKeeper())
            camera = client.camera
            try:
                if cid in new:
                    keeper.frame(camera, robot + behind, robot)
                elif aim and tracking:
                    keeper.frame(camera, robot + camera.position - camera.look_at,
                                 robot)
                elif tracking and ray is not None:
                    keeper.step(camera, self.scene_offset, self.scene_yaw, ray, dt)
            except AssertionError:   # no camera report from the page yet
                if cid in new:
                    with self._lock:
                        self._to_frame.add(cid)


__all__ = ["OperatorPanel", "SightKeeper", "Tracker", "add_depth_camera",
           "base_yaw", "depth_rgb", "march_status", "z_rotation"]
