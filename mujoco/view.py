#!/usr/bin/env python3
"""Light host: policy-driven robot + viser web viewer, keyboard/gamepad drive.

Same sim core as physics_view (50 Hz policy). Default presentation is the
headless viser scene (--view web): no DISPLAY/GLFW dependency, an order of
magnitude cheaper than the native viewer. Its operator panel (webpanel.py,
top-right corner of the page) picks the robot (sim/robots.py: switching
reloads the body and its policy), toggles the perception policy (depth;
off = the robot's blind policy, once it has one) and the gamepad, keeps
the view on the robot and turned with it (the browser's orbit/zoom/pan
circle it as it walks), shows the policy's depth input, and carries the Reset/Stop
buttons. The dog-eye splat
is NOT rendered by default -- `--preview` builds the wgpu splat pipeline
and serves it as an MJPEG page for a second screen:

  scene:  http://127.0.0.1:8012/    (viser, when --view web)
  dog:    http://127.0.0.1:8888/    (--preview only; --p-host/--p-port)

Drive with the global arrows (hold = move; on release the source flushes
zeros for 1 s and then goes idle -- the gazebo teleop lifecycle, see
sim/keyboard.py), `,`/`.` strafe; a gamepad on the sim host drives when
gamepad control is on (`--gamepad`, the panel checkbox or A on the pad;
sim/gamepad.py), held arrows still win over it.

Policies: the robot's registry policy (`--robot`, default go2: the
go2_depth_v2 student; `--blind` starts on the blind one); `--policy <file>`
pins one ONNX for the robot. `--view glfw` restores the native MuJoCo
window (with the 12 s wedge guard).

Clean exit path uses os._exit: the mixed GL atexit teardown segfaults on
this driver (kernel error 15 after DONE).
"""
from __future__ import annotations

import copy
import ctypes
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

try:
    ctypes.CDLL("libX11.so.6").XInitThreads()
except OSError:
    pass

import cv2
cv2.setNumThreads(0)   # OpenCV pool busy-spins; keep it serial
try:
    import ctypes as _ct
    _libc = _ct.CDLL("libc.so.6", use_errno=True)
    _libc.prctl(15, _ct.c_char_p(b"splatsense-view"), 0, 0, 0)
except Exception:     # noqa: BLE001 -- cosmetic only
    pass
import mujoco
import mujoco.viewer
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sim.accel import AccelLimit  # noqa: E402
from sim.gamepad import GamepadCmd, GamepadOverride  # noqa: E402
from sim.guard import WallGuard  # noqa: E402
from sim.keyboard import ConstantCmd, HoldCmd, KeyboardOverride  # noqa: E402
from sim.plant import (  # noqa: E402
    SCENES,
    build_model,
    get_scene,
    reset_to_spawn,
    web_view_model,
)
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.robots import DEFAULT_ROBOT, ROBOTS, get_robot  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402
from sim.thermal import HipThermal  # noqa: E402

CONTROL_DT = 0.02
RENDER_HZ = 15.0   # --preview: splat render + MJPEG push
WEB_HZ = 50.0      # web viewer only: scene + robot tracking every control tick
CAM = "infra1"


# --------------------------------------------------------------------- #
# MJPEG server: / (page) + /stream (multipart) + /frame.jpg (snapshot)
# --------------------------------------------------------------------- #

class MjpegServer:
    def __init__(self, host: str, port: int) -> None:
        self._lock = threading.Lock()
        self._jpeg = b""
        self._seq = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # noqa: N802 -- silence
                pass

            def do_GET(self):            # noqa: N802
                if self.path in ("/", "/index.html"):
                    body = b"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pie light</title>
<style>
  html,body{margin:0;height:100%;background:#111;overflow:hidden}
  body{display:flex;flex-direction:column}
  #bar{flex:0 0 auto;color:#9c9;font:12px monospace;padding:4px 8px}
  #wrap{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
  img.fit{max-width:100%;max-height:100%;object-fit:contain}
  img.native{max-width:none;max-height:none}
  #wrap.pad{padding:0 8px 8px}
</style></head>
<body>
<div id="bar">pie light &mdash; click image: fit &harr; 1:1 &nbsp;|&nbsp; <a style="color:#9c9" href="/frame.jpg">snapshot</a></div>
<div id="wrap" class="pad"><img id="v" class="fit" src="/stream"></div>
<script>
document.getElementById("v").addEventListener("click", function () {
  this.classList.toggle("fit"); this.classList.toggle("native");});
</script>
</body></html>"""
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = -1
                    while True:
                        with outer._lock:
                            jpeg, seq = outer._jpeg, outer._seq
                        if jpeg == b"":
                            time.sleep(0.05)
                            continue
                        if seq == last:
                            time.sleep(1.0 / RENDER_HZ)
                            continue
                        last = seq
                        try:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(jpeg)).encode()
                                + b"\r\n\r\n" + jpeg + b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            return
                elif self.path == "/frame.jpg":
                    with outer._lock:
                        jpeg = outer._jpeg
                    if jpeg == b"":
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                else:
                    self.send_error(404)

        self._srv = ThreadingHTTPServer((host, port), Handler)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def push(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1

    def close(self) -> None:
        self._srv.shutdown()


def depth_cmap(depth, vmin=200.0, vmax=4000.0):
    d = np.clip(depth.astype(np.float32), vmin, vmax)
    d = ((d - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


def _argv_value(flag: str, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# build_model once per (scene, robot): a robot switch then copies a compiled
# model (~30 ms) instead of compiling it again (up to 0.3 s). Every World gets
# its own copy -- worlds and their web scenes mutate theirs. _MESHES holds the
# web scene's merged body meshes per (scene, robot), keyed by the model's ids.
_MODELS: dict[tuple[str, str], tuple] = {}
_MESHES: dict[tuple[str, str], dict] = {}


def _compiled(scene_name: str, robot: str):
    key = (scene_name, robot)
    if key not in _MODELS:
        _MODELS[key] = build_model(scene_name, robot)
    model, b = _MODELS[key]
    return copy.deepcopy(model), b        # Bindings is frozen: shared


_BOX_CORNERS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1)
                         for z in (-1, 1)], dtype=np.float64)


def robot_length(model, data, b) -> float:
    """Nose-to-tail span (m) of the robot's shown geoms along its heading, in
    the current pose: the web view frames it from a multiple of it."""
    mujoco.mj_kinematics(model, data)
    joint = int(np.flatnonzero(model.jnt_qposadr == b.root_qpos_adr)[0])
    heading = data.xmat[model.jnt_bodyid[joint]].reshape(3, 3)[:, 0].copy()
    heading[2] = 0.0
    heading /= np.linalg.norm(heading)
    shown = ((model.body_weldid[model.geom_bodyid] != 0)
             & (model.geom_group < 3) & (model.geom_rgba[:, 3] > 0))
    lo, hi = np.inf, -np.inf
    for g in np.flatnonzero(shown):
        corners = (_BOX_CORNERS * model.geom_aabb[g, 3:] + model.geom_aabb[g, :3]
                   ) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        along = corners @ heading
        lo, hi = min(lo, along.min()), max(hi, along.max())
    return float(hi - lo)


class World:
    """One robot in the scene: plant, policy, depth camera, runtime. The
    depth camera (EGL) is opened by start() -- after the splat pipeline's
    wgpu init on first boot (see main). `start` = (pos, yaw_deg) stands the
    robot there instead of at the spawn (a robot switch keeps the place);
    Reset still goes back to pos/yaw_deg."""

    def __init__(self, scene_name: str, robot: str, kind: str, cmd, pos,
                 yaw_deg: float, policy_path: str | None = None,
                 wall_alpha: float = 1.0, start=None) -> None:
        self.robot = get_robot(robot)
        self.scene_name, self.pos, self.yaw_deg, self.cmd = scene_name, pos, yaw_deg, cmd
        self.model, self.b = _compiled(scene_name, robot)
        self.model.vis.quality.shadowsize = 0          # shadows off (viewer + renders)
        # shaft walls: BLEND alpha stacks PER-PANEL now (webviewer emits
        # translucent geoms as separate meshes), so the XML's 0.1 alpha
        # renders as real see-through glass; stairs, landings and the
        # dog stay opaque so the structure reads.
        walls = ((self.model.geom_bodyid == 0)
                 & (self.model.geom_rgba[:, 3] < 0.5))
        self.model.geom_rgba[walls, 3] = wall_alpha
        self.data = mujoco.MjData(self.model)
        start_pos, start_yaw = start if start is not None else (pos, yaw_deg)
        reset_to_spawn(self.model, self.data, self.b, settle=True,
                       pos=start_pos, yaw_deg=start_yaw)
        self.length = robot_length(self.model, self.data, self.b)
        # what the web view shows opaque (its colour rule: material rgba
        # first): static geoms of the shown groups 0-2, walls at
        # wall_alpha < 0.5 excluded -- the surfaces ray() stops at
        m = self.model
        alpha = m.geom_rgba[:, 3].copy()
        alpha[m.geom_matid >= 0] = m.mat_rgba[m.geom_matid[m.geom_matid >= 0], 3]
        self._opaque = ((m.body_weldid[m.geom_bodyid] == 0)
                        & (m.geom_group < 3) & (alpha >= 0.5))
        self._ray_geom = np.zeros(1, np.int32)
        self.kind = kind
        self.policy = self._load_policy(kind, policy_path)
        self.policy_path = policy_path   # a --policy override (None: the registry's)
        self.cam = None
        self.rt = None

    def ray(self, origin, direction, reach: float):
        """Distance (m) along the unit `direction` from `origin` to the first
        surface the web view shows opaque, or None within `reach`; the robot
        and the see-through walls are looked past."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        t = 0.0
        for _ in range(32):
            h = mujoco.mj_ray(self.model, self.data, origin + t * direction,
                              direction, None, 1, -1, self._ray_geom)
            if h < 0 or t + h >= reach:
                return None
            t += h
            if self._opaque[self._ray_geom[0]]:
                return t
            t += 1e-4
        return None

    def _load_policy(self, kind: str, policy_path: str | None):
        if policy_path is None:
            path = self.robot.policy_path(kind)
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"{self.robot.label} has no {kind} policy "
                    f"({path.name if path else 'none registered'})")
        else:
            path = Path(policy_path)
        self.policy_name = path.name
        return PieOnnxPolicy(path, provider="cpu")

    def start(self) -> None:
        self.cam = DepthCamera(self.model, self.b.depth_camera_id)
        self.rt = PieRuntime(self.model, self.b, self.data, self.policy, self.cam,
                             self.cmd, spawn_pos=self.pos,
                             spawn_yaw_deg=self.yaw_deg)

    def set_kind(self, kind: str) -> None:
        """Swap perception <-> blind in place (robot state kept, fresh
        policy memory and histories)."""
        policy = self._load_policy(kind, None)
        old = self.rt
        self.rt = PieRuntime(self.model, self.b, self.data, policy, self.cam,
                             self.cmd, spawn_pos=self.pos,
                             spawn_yaw_deg=self.yaw_deg)
        if old is not None:
            old.close()
        self.policy, self.kind, self.policy_path = policy, kind, None

    @property
    def depth_spec(self):
        """The driving policy's depth frame (contract_v2.DepthSpec; None when
        blind)."""
        return self.rt.depth_spec if self.rt is not None and self.rt.uses_depth else None

    def reset(self) -> None:
        reset_to_spawn(self.model, self.data, self.b, settle=True,
                       pos=self.pos, yaw_deg=self.yaw_deg)

    def close(self) -> None:
        if self.rt is not None:
            self.rt.close()
        if self.cam is not None:
            self.cam.close()

    def pad_envelope(self):
        """Gamepad full deflection (and the keyboard's range) = the policy's
        training command envelope."""
        return self.policy.contract.COMMAND_LOW, self.policy.contract.COMMAND_HIGH


def main() -> int:
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = pos_args[0] if pos_args else "2060"
    scene_name = _argv_value("--scene", "map3")
    sc = get_scene(scene_name)
    spawn = _argv_value("--spawn", sc.default_spawn)
    robot_name = _argv_value("--robot", DEFAULT_ROBOT)
    kind = "blind" if "--blind" in sys.argv else "perception"
    if not get_robot(robot_name).has_policy(kind):
        # Same fallback as switching robots in the panel.
        kind = "blind" if kind == "perception" else "perception"
    policy_path = _argv_value("--policy", None)
    view_mode = _argv_value("--view", "web")    # web | glfw
    web_port = int(_argv_value("--web-port", "8012"))
    wall_alpha = float(_argv_value("--wall-alpha", "0.1"))
    preview = "--preview" in sys.argv
    use_gamepad = "--gamepad" in sys.argv
    # MJPEG bind, preview only: --p-host/--p-port (--host/--port aliases)
    host = _argv_value("--p-host", _argv_value("--host", "127.0.0.1"))
    port = int(_argv_value("--p-port", _argv_value("--port", "8888")))
    scale = float(_argv_value("--scale", "1.0"))   # composite downscale before JPEG
    q = int(_argv_value("--q", "80"))              # JPEG quality
    if spawn not in sc.spawns:
        raise SystemExit(f"unknown spawn {spawn!r}; choose {sorted(sc.spawns)}")
    pos, yaw_deg = sc.spawns[spawn]

    # Command sources: held arrows win (KeyboardOverride), then the gamepad
    # while its control is on, else zeros; AccelLimit ramps what they ask for
    # (sim/accel.py; the pad's B stops at once) and WallGuard stops it in
    # front of walls (sim/guard.py; `world` is read when it runs, so it follows
    # switches). Reset/Stop ride kb.key (82 / 32) -- the codes the native
    # viewer key_callback would receive.
    kb = HoldCmd()
    pad = GamepadCmd(enabled=use_gamepad)
    mux = KeyboardOverride(kb, GamepadOverride(pad, ConstantCmd(vx=0.0)))
    accel = AccelLimit(mux, brake=lambda: pad.stop_held)
    cmd = WallGuard(accel, lambda: world.rt)
    world = World(scene_name, robot_name, kind, cmd, pos, yaw_deg,
                  policy_path=policy_path,
                  wall_alpha=wall_alpha if view_mode == "web" else 1.0)
    pad.set_envelope(*world.pad_envelope())
    kb.set_envelope(*world.pad_envelope())
    # the hips' estimated motor temperatures (panel plot), restarted per robot
    thermal = HipThermal(world.data, world.b, world.robot.name)

    # wgpu splat pipeline (--preview only, its one consumer): static scene
    # only, exactly like hil -- GaussianCloud.from_ply -> RAW INRIA values
    # activated before the pipeline. The render cam is the infra1 dog-eye
    # rig and never sees the dog's own body, so there are no dog gaussians
    # to animate (the old w1_static merged cloud paid projection time for
    # 606k of them); the dog itself is drawn by the mesh renderers (mujoco
    # viewer / webviewer).
    pipe = ad = None
    variant = "-"
    if preview:
        from gausscam.adapters.mujoco import MuJoCoAdapter
        from gausscam.backends.webgpu import Pipeline, pick_variant
        from gausscam.core.assets import GaussianCloud
        from hil import CONTRACT_FOVY_DEG, CONTRACT_H, CONTRACT_W
        ad = MuJoCoAdapter(world.model, world.data)
        cpos, cxm = ad.camera_poses([CAM])
        _scene_ply = _HERE.parent / "model" / "splat" / sc.ply
        if not _scene_ply.is_file():
            raise SystemExit(
                f"splat scene missing: {_scene_ply}\n"
                "unzip model.zip at the repo root (creates model/splat/)")
        cloud = GaussianCloud.from_ply(_scene_ply)
        d = {
            "xyz": cloud.xyz,
            "rot": cloud.rot,
            "scale": np.exp(cloud.scale).astype(np.float32),
            "opacity": (1.0 / (1.0 + np.exp(-cloud.opacity))).astype(np.float32),
            "sh": np.ascontiguousarray(cloud.sh[:, :3]),
            "slots": np.zeros(0, np.int32),
            "W": np.int32(CONTRACT_W),
            "H": np.int32(CONTRACT_H),
            "fovy": np.float32(CONTRACT_FOVY_DEG),
            "cam_pos": cpos.astype(np.float32),
            "cam_xmat": cxm.astype(np.float32),
        }
        variant = pick_variant(sel)
        pipe = Pipeline(d, sel, variant=variant)
        print(f"device: {pipe.adapter.info['device']}  variant={variant}",
              flush=True)
    # DepthCamera AFTER wgpu: on some hybrid-graphics NVIDIA drivers (2060M
    # + 595) an EGL context created before wgpu init dies once Vulkan is up.
    world.start()
    print(f"robot={world.robot.name} policy={world.policy_name} "
          f"contract={world.policy.contract.name} spawn={spawn}", flush=True)

    srv = MjpegServer(host, port) if preview else None
    if srv is not None:
        print(f"preview: http://{host}:{port}/  "
              f"(snapshot: /frame.jpg)  jpeg scale={scale} q={q}", flush=True)

    # web viewer: the DEFAULT presentation -- headless viser scene, no
    # DISPLAY/GLFW dependency, ~1/17 the CPU of the native one, with the
    # operator panel (webpanel.py) above mjviser's tabs.
    web_server = web_scene = panel = tabs = None
    if view_mode == "web":
        try:
            import viser
            from webpanel import (OperatorPanel, add_depth_camera, base_yaw, guard_status,
                                  march_status, sync_depth_camera)
            from webviewer import ViserMujocoScene
            web_server = viser.ViserServer(port=web_port)

            def _on_button(name: str) -> None:
                kb.key({"reset": 82, "stop": 32}[name])

            # Map dropdown: the scene registry; not with --preview (its splat
            # pipeline holds the boot map's cloud)
            panel = OperatorPanel(
                web_server, ROBOTS, world.robot.name, gamepad=pad, keyboard=kb,
                accel=accel, guard=cmd, on_button=_on_button, scene=scene_name,
                scenes=None if preview else {n: s.label or n for n, s in SCENES.items()})

            def _build_web_scene():
                web_server.scene.reset()
                # its own model copy: the navcore unit drawn on the head,
                # the depth camera's model untouched
                scene = ViserMujocoScene(
                    web_server, web_view_model(world.model, world.robot), num_envs=1,
                    mesh_cache=_MESHES.setdefault((scene_name, world.robot.name), {}))
                # mjviser's own tracking (the raw base pinned to the origin,
                # its "Track camera" checkbox) off: the panel's Track robot
                # shifts and turns the scene itself, by the low-passed base
                # and heading.
                scene.camera_tracking_enabled = False
                scene.scene_offset = panel.scene_offset
                scene.scene_yaw = panel.scene_yaw
                # The panel frames new pages too (behind the robot), so a
                # switch leaves views alone.
                scene.frame_clients = False
                scene.show_contact_points = True
                # hulls render at 0.5 opacity over EVERY geom and read
                # opaque; collision viewing is covered by the G3 group
                # toggle instead.
                scene.show_convex_hull = False
                # the policy's depth camera, a small pyramid (depth policy only)
                add_depth_camera(scene, world.length, world.depth_spec)
                scene.show_camera_frustums = world.kind == "perception"
                scene.update_from_mjdata(world.data)
                return scene

            web_scene = _build_web_scene()
            tabs = web_scene.create_visualization_gui()
            panel.show_robot(world.robot.name, world.kind == "perception",
                             world.robot.has_policy("perception"),
                             world.robot.has_policy("blind"), world.length)
            # viser silently bumps to the next free port when busy -- print
            # the ACTUAL bound port, not the request.
            print(f"[WEBVIEWER] http://0.0.0.0:{web_server.get_port()}  "
                  f"(wall alpha={wall_alpha}, operator panel: map/robot/"
                  "perception/gamepad/track robot, Reset/Stop)", flush=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[WEBVIEWER] init failed: {exc}", flush=True)
            web_scene = panel = None
        if panel is not None:
            # Compile and merge the other robots now, before the loop runs,
            # so a switch does not pay for it on the control thread.
            t0 = time.perf_counter()
            try:
                for name, r in ROBOTS.items():
                    if name != world.robot.name and not r.missing():
                        m, _ = _compiled(scene_name, name)
                        ViserMujocoScene.fill_mesh_cache(
                            web_view_model(m, r), _MESHES.setdefault((scene_name, name), {}))
                print(f"[WEBVIEWER] switch cache: {len(_MODELS)} robots in "
                      f"{time.perf_counter() - t0:.1f} s", flush=True)
            except Exception as exc:      # noqa: BLE001 -- switches still work, slower
                print(f"[WEBVIEWER] switch cache failed: {exc}", flush=True)

    def _switch(request) -> None:
        """Apply a panel request on the control thread (between ticks)."""
        nonlocal world, web_scene, tabs, ad, thermal, scene_name, sc, spawn, pos, yaw_deg
        what, value = request
        try:
            if what == "perception":
                kind_new = "perception" if value else "blind"
                if kind_new != world.kind:
                    world.set_kind(kind_new)
                    if world.depth_spec is not None:
                        # a navcore student moves the camera: redraw its pyramid
                        sync_depth_camera(web_scene, world.model, world.depth_spec, world.length)
            elif what == "scene":
                if value != scene_name:
                    # the same robot and policy in the new map, at its spawn
                    t0 = time.perf_counter()
                    sc_new = get_scene(value)
                    pos_new, yaw_new = sc_new.spawns[sc_new.default_spawn]
                    new = World(value, world.robot.name, world.kind, cmd, pos_new, yaw_new,
                                policy_path=world.policy_path, wall_alpha=wall_alpha)
                    new.start()
                    world.close()
                    world = new
                    scene_name, sc, spawn = value, sc_new, sc_new.default_spawn
                    pos, yaw_deg = pos_new, yaw_new
                    thermal = HipThermal(world.data, world.b, world.robot.name)
                    panel.reset_temperatures()
                    with web_server.atomic():
                        web_scene = _build_web_scene()
                        tabs.remove()
                        tabs = web_scene.create_visualization_gui()
                    # the robot is somewhere else now: every page behind it again
                    panel.reframe()
                    print(f"[SWITCH] map {scene_name} built in "
                          f"{time.perf_counter() - t0:.2f} s", flush=True)
            elif value != world.robot.name:
                # keep the perception choice when the new robot has it
                robot = get_robot(value)
                kind_new = (world.kind if robot.has_policy(world.kind)
                            else "perception" if robot.has_policy("perception")
                            else "blind")
                # the new robot stands where the old one is, facing the same
                # way, so the (tracked) view stays put
                t0 = time.perf_counter()
                root = world.b.root_qpos_adr
                here = tuple(float(v) for v in world.data.qpos[root:root + 3])
                heading = math.degrees(base_yaw(world.data.qpos[root + 3:root + 7]))
                new = World(scene_name, value, kind_new, cmd, pos, yaw_deg,
                            wall_alpha=wall_alpha, start=(here, heading))
                new.start()
                world.close()
                world = new
                thermal = HipThermal(world.data, world.b, world.robot.name)
                panel.reset_temperatures()
                # one atomic update: no frame of empty scene between robots
                with web_server.atomic():
                    web_scene = _build_web_scene()
                    tabs.remove()
                    tabs = web_scene.create_visualization_gui()
                print(f"[SWITCH] built in {time.perf_counter() - t0:.2f} s",
                      flush=True)
                if pipe is not None:
                    ad = MuJoCoAdapter(world.model, world.data)
            print(f"[SWITCH] map={scene_name} robot={world.robot.name} kind={world.kind} "
                  f"policy={world.policy_name} "
                  f"contract={world.policy.contract.name}", flush=True)
        except Exception as exc:      # noqa: BLE001 -- keep driving the old one
            print(f"[SWITCH] {request} refused: {exc}", flush=True)
            for client in web_server.get_clients().values():
                client.add_notification(title="Switch refused", body=str(exc),
                                        auto_close_seconds=8.0)
        pad.set_envelope(*world.pad_envelope())
        kb.set_envelope(*world.pad_envelope())
        panel.show_robot(world.robot.name, world.kind == "perception",
                         world.robot.has_policy("perception"),
                         world.robot.has_policy("blind"), world.length, scene=scene_name)

    # native GLFW viewer (--view glfw only): 12 s wedge guard, headless
    # continuation on timeout (web viewer covers the no-display case).
    viewer = None
    use_mjv = False
    if view_mode == "glfw":
        holder: dict = {}
        ready = threading.Event()

        def _open_viewer():
            try:
                ui = os.environ.get("MJUI", "both")   # both | left | right | none
                holder["v"] = mujoco.viewer.launch_passive(
                    world.model, world.data, key_callback=kb.key,
                    show_left_ui=ui in ("both", "left"),
                    show_right_ui=ui in ("both", "right"))
            except Exception as exc:      # noqa: BLE001
                print(f"[VIEWER] launch failed: {exc}", flush=True)
            finally:
                ready.set()

        threading.Thread(target=_open_viewer, daemon=True).start()
        if not ready.wait(timeout=12.0):
            print("[VIEWER] window not mapped in 12s (compositor wedged?) "
                  "-- continuing headless; web viewer is not started in "
                  "glfw mode (restart with --view web)", flush=True)
        else:
            viewer = holder.get("v")
            use_mjv = viewer is not None
            if viewer is not None:
                try:   # viewer UI keeps its own shadow flag; force it off too
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                except Exception:      # noqa: BLE001
                    pass

    quit_req = False
    # marching in place: M / pad X-Y toggle it, Stop and Reset end it; the
    # policy marches on a zero command if it was trained to (gait_march)
    march = False
    fps = 0.0
    n_frames = 0
    t_last = time.perf_counter()
    t_view = t_last
    period = 1.0 / (RENDER_HZ if preview else WEB_HZ)
    next_ctrl = time.perf_counter()
    next_render = next_ctrl

    while (not use_mjv or viewer.is_running()) and not quit_req:
        now = time.perf_counter()
        if now >= next_ctrl:
            request = panel.take_request() if panel is not None else None
            if request is not None:
                _switch(request)
            if kb.consume_reset():
                world.reset()
                cmd.reset()
                march = False
                print("[RESET] back to spawn", flush=True)
            if kb.consume_march() | pad.consume_march():
                march = not march
                print(f"[MARCH] {'on' if march else 'off'}", flush=True)
            if (kb.consume_stop() or pad.stop_held) and march:
                march = False
                print("[MARCH] off (stop)", flush=True)
            world.rt.march = march
            world.rt.tick()
            thermal.step(CONTROL_DT)
            if panel is not None:
                panel.plot_temperatures(thermal.time, thermal.labels,
                                        thermal.temperatures)
            next_ctrl = max(next_ctrl + CONTROL_DT, time.perf_counter())
        if now >= next_render:
            rt = world.rt
            if pipe is not None:
                cp, cm = ad.camera_poses([CAM])   # rig rides the base
                pipe.set_cam(cp, cm)
                rgb_p, dep_p = pipe.render_frame()
                rgb, depth = pipe.unpack(rgb_p, dep_p)

            if srv is not None:
                rgb_l = rgb[0][:, :, ::-1].copy()
                dep_l = depth_cmap(depth[0])
                bar = np.full((26, rgb_l.shape[1], 3), 24, np.uint8)
                cv2.putText(bar,
                            f"{world.robot.label} {variant} {fps:4.1f}fps "
                            f"vx={rt.command[0]:+.2f} vy={rt.command[1]:+.2f} "
                            f"wz={rt.command[2]:+.2f} "
                            f"cmd={mux.state:5s} "
                            f"up={rt.base_uprightness():.2f} "
                            f"phy={rt.physics_ms:.1f} pol={rt.policy_ms:.1f}ms "
                            f"xyz={np.round(rt.base_pos(), 1)}",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                bar2 = np.full((24, rgb_l.shape[1], 3), 24, np.uint8)
                cv2.putText(bar2,
                            "GLOBAL arrows: hold Up/Down vx +0.5/-0.2  "
                            "Left/Right wz +/-0.3  ,/. vy  ·  release: 1s "
                            "zero-flush then idle",
                            (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (160, 220, 160), 1, cv2.LINE_AA)
                img = cv2.vconcat([bar, rgb_l, bar2, dep_l])
                if scale != 1.0:
                    img = cv2.resize(img, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", img,
                                        [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    srv.push(jpeg.tobytes())

            # web viewer refresh reads the LIVE mjData -> control thread
            # only (same race family as mj_copyData); throttled at the
            # render cadence, never from the render thread.
            # half a period of slack: the block is paced by next_render already,
            # and a strict >= period skips every frame that wakes earlier than
            # the last one did (sleep jitter), ~1/3 of them
            if web_scene is not None and now - t_view >= 0.5 * period:
                root = world.b.root_qpos_adr
                pad_txt = ("开" if pad.enabled else "关") if pad.connected else "未连"
                web_scene.scene_offset, web_scene.scene_yaw = panel.update(
                    rt.base_pos(), world.data.qpos[root + 3:root + 7],
                    rt.depth_frame(),
                    f"{world.policy_name} · cmd {rt.command[0]:+.2f} "
                    f"{rt.command[1]:+.2f} {rt.command[2]:+.2f} · 手柄{pad_txt}"
                    + march_status(march, world.policy.contract) + guard_status(cmd),
                    ray=world.ray)
                web_scene.show_camera_frustums = world.kind == "perception"
                web_scene.update_from_mjdata(world.data)
                t_view = now

            if use_mjv:
                viewer.sync()
            n_frames += 1
            now2 = time.perf_counter()
            if now2 - t_last >= 1.0:
                fps = n_frames / (now2 - t_last)
                n_frames = 0
                t_last = now2
                cam_txt = ""
                try:   # first connected web client's reported camera pose
                    clients = web_server.get_clients()
                    if clients:
                        cam = next(iter(clients.values())).camera
                        p, l = cam.position, cam.look_at
                        cam_txt = (f" cam=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}"
                                   f">@({l[0]:.1f},{l[1]:.1f},{l[2]:.1f})")
                    else:
                        cam_txt = " cam=none"
                except Exception as exc:      # noqa: BLE001
                    cam_txt = f" camERR={exc!r}"
                print(f"{fps:4.1f}fps  cmd={mux.state:5s} "
                      f"{world.robot.name} "
                      f"vx={rt.command[0]:+.2f} vy={rt.command[1]:+.2f} "
                      f"wz={rt.command[2]:+.2f} "
                      f"up={rt.base_uprightness():.2f} "
                      f"xyz={np.round(rt.base_pos(), 1)}{cam_txt}", flush=True)
            next_render += period

        wake = min(next_ctrl, next_render) - time.perf_counter()
        if wake > 0:
            time.sleep(wake)

    kb.stop()
    pad.stop()
    world.close()
    if srv is not None:
        srv.close()
    print("bye", flush=True)
    os._exit(0)   # mixed GL atexit teardown segfaults on this driver


if __name__ == "__main__":
    sys.exit(main())
