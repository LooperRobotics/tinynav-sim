#!/usr/bin/env python3
"""Light host: PIE policy + wgpu splat + viser web viewer, keyboard drive.

Same sim/render core as physics_view (50 Hz policy, 15 Hz splat). Default
presentation is the headless viser scene (--view web): no DISPLAY/GLFW
dependency, an order of magnitude cheaper than the native viewer, with
Reset/Stop buttons in the browser. The dog-eye image is NOT streamed by
default — `--preview` serves it as an MJPEG page for a second screen:

  scene:  http://127.0.0.1:8012/    (viser, when --view web)
  dog:    http://127.0.0.1:8888/    (--preview only; --p-host/--p-port)

Drive with the global arrows (hold = move; on release the source flushes
zeros for 1 s and then goes idle — the gazebo teleop lifecycle, see
sim/keyboard.py). Dual-expert selection is AUTO by default: backward
commands ride the flat omni expert, everything else the stairs veteran —
the unified-training campaign showed the two skills cannot share one net,
so deployment switches between them (see sim/expert.py). `--expert
stairs|flat` forces one side of the pair, `--expert single --policy <file>`
pins one ONNX. `--view glfw` restores the native MuJoCo window (with
the 12 s wedge guard).

Clean exit path uses os._exit: the mixed GL atexit teardown segfaults on
this driver (kernel error 15 after DONE).
"""
from __future__ import annotations

import ctypes
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

from sim.keyboard import HoldCmd  # noqa: E402
from sim.plant import (  # noqa: E402
    SCENES,
    build_model,
    get_scene,
    reset_to_spawn,
)
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402

ONNX = _HERE / "assets" / "policy" / "policy.onnx"
FLAT_ONNX = _HERE / "assets" / "policy" / "policy_24000_flat_omni.onnx"
CONTROL_DT = 0.02
RENDER_HZ = 15.0
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


def main() -> int:
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = pos_args[0] if pos_args else "2060"
    scene_name = _argv_value("--scene", "map3")
    sc = get_scene(scene_name)
    spawn = _argv_value("--spawn", sc.default_spawn)
    expert_mode = _argv_value("--expert", "auto")   # auto | stairs | flat | single
    policy_path = (sys.argv[sys.argv.index("--policy") + 1]
                   if "--policy" in sys.argv else None)
    view_mode = _argv_value("--view", "web")    # web | glfw
    web_port = int(_argv_value("--web-port", "8012"))
    wall_alpha = float(_argv_value("--wall-alpha", "0.1"))
    preview = "--preview" in sys.argv
    # MJPEG bind, preview only: --p-host/--p-port (--host/--port aliases)
    host = _argv_value("--p-host", _argv_value("--host", "127.0.0.1"))
    port = int(_argv_value("--p-port", _argv_value("--port", "8888")))
    scale = float(_argv_value("--scale", "1.0"))   # composite downscale before JPEG
    q = int(_argv_value("--q", "80"))              # JPEG quality
    if spawn not in sc.spawns:
        raise SystemExit(f"unknown spawn {spawn!r}; choose {sorted(sc.spawns)}")
    pos, yaw_deg = sc.spawns[spawn]

    model, b = build_model(scene_name)
    model.vis.quality.shadowsize = 0          # shadows off (viewer + renders)
    data = mujoco.MjData(model)
    reset_to_spawn(model, data, b, settle=True, pos=pos, yaw_deg=yaw_deg)

    if policy_path is not None or expert_mode == "single":
        from sim.policy import PieOnnxPolicy
        policy = PieOnnxPolicy(Path(policy_path) if policy_path else ONNX,
                               provider="cpu")
    else:
        from sim.expert import DualExpertPolicy
        policy = DualExpertPolicy(ONNX, FLAT_ONNX, mode=expert_mode,
                                  provider="cpu")

    def expert_tag() -> str:
        return str(getattr(policy, "active_name", "single"))   # single-threaded session
    cmd = HoldCmd()

    # wgpu splat pipeline: static map3 scene only, exactly like hil --
    # GaussianCloud.from_ply -> RAW INRIA values activated before the
    # pipeline. The render cam is the infra1 dog-eye rig and never sees the
    # dog's own body, so there are no dog gaussians to animate (the old
    # w1_static merged cloud paid projection time for 606k of them); the
    # dog itself is drawn by the mesh renderers (mujoco viewer / webviewer).
    from gausscam.adapters.mujoco import MuJoCoAdapter
    from gausscam.backends.webgpu import Pipeline, pick_variant
    from gausscam.core.assets import GaussianCloud
    from hil import CONTRACT_FOVY_DEG, CONTRACT_H, CONTRACT_W
    ad = MuJoCoAdapter(model, data)
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
    # DepthCamera AFTER wgpu: on some hybrid-graphics NVIDIA drivers (2060M
    # + 595) an EGL context created before wgpu init dies once Vulkan is up.
    cam = DepthCamera(model, b.depth_camera_id)
    rt = PieRuntime(model, b, data, policy, cam, cmd,
                    spawn_pos=pos, spawn_yaw_deg=yaw_deg)

    srv = MjpegServer(host, port) if preview else None
    if srv is not None:
        print(f"preview: http://{host}:{port}/  "
              f"(snapshot: /frame.jpg)  jpeg scale={scale} q={q}", flush=True)
    print(f"device: {pipe.adapter.info['device']}  variant={variant}  "
          f"spawn={spawn}", flush=True)

    # web viewer: the DEFAULT presentation -- headless viser scene, no
    # DISPLAY/GLFW dependency, ~1/17 the CPU of the native one. Reset/Stop
    # buttons ride cmd.key (82 / 32) -- the same codes the native viewer
    # key_callback would receive, so reset/stop behave identically.
    web_scene = None
    if view_mode == "web":
        try:
            import viser
            from webviewer import ViserMujocoScene
            # shaft walls: BLEND alpha stacks PER-PANEL now (webviewer emits
            # translucent geoms as separate meshes), so the XML's 0.1 alpha
            # renders as real see-through glass; stairs, landings and the
            # dog stay opaque so the structure reads.
            walls = ((model.geom_bodyid == 0)
                     & (model.geom_rgba[:, 3] < 0.5))
            model.geom_rgba[walls, 3] = wall_alpha
            web_server = viser.ViserServer(port=web_port)
            web_scene = ViserMujocoScene(web_server, model, num_envs=1)
            # "Track camera" force-aims every client at the world ORIGIN --
            # the dog lives at the spawn, so tracking off + framing below.
            web_scene.camera_tracking_enabled = False
            web_scene.create_visualization_gui()

            # Camera framing: ONCE, for the FIRST client only. Embedded
            # browsers drop+reconnect the websocket under sustained scene
            # updates (observed: connection churn exactly when the dog
            # starts moving), and every reconnect fires on_client_connect —
            # per-connect framing would yank the camera back to spawn each
            # time the operator drives. After the first framing the camera
            # belongs to the operator, full stop.
            cam_framing = {"armed": True}

            def _apply_framing(client):
                if not cam_framing["armed"]:
                    return
                try:
                    p = np.asarray(pos, float)
                    # spawn sits in a ~0.9 m wide shaft compartment
                    # (x wall 0.44 m away): stay inside it and look
                    # from the shaft-center side.
                    client.camera.position = p + np.array(
                        [-0.35, -1.6, 0.6])
                    client.camera.look_at = p
                except Exception:      # noqa: BLE001
                    pass

            def _frame_spawn(client):
                if not cam_framing["armed"]:
                    return
                cam_framing["armed"] = False   # this client owns the framing
                _apply_framing(client)
                # one late re-apply: the client re-frames itself once the
                # scene upload lands, overwriting a single set at connect
                threading.Timer(2.0, lambda: _apply_framing(client)).start()

            web_server.on_client_connect(_frame_spawn)

            def _add_cmd_button(label: str, keycode: int) -> None:
                btn = web_server.gui.add_button(label)

                @btn.on_click
                def _(_event) -> None:
                    cmd.key(keycode)

            _add_cmd_button("Reset to spawn", 82)
            _add_cmd_button("Stop", 32)
            web_scene.show_contact_points = True
            # hulls render at 0.5 opacity over EVERY geom and read opaque;
            # collision viewing is covered by the G3 group toggle instead.
            web_scene.show_convex_hull = False
            web_scene.update_from_mjdata(data)
            # viser silently bumps to the next free port when busy -- print
            # the ACTUAL bound port, not the request.
            print(f"[WEBVIEWER] http://0.0.0.0:{web_server.get_port()}  "
                  f"(wall alpha={wall_alpha}, GUI: Reset/Stop)", flush=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[WEBVIEWER] init failed: {exc}", flush=True)
            web_scene = None

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
                    model, data, key_callback=cmd.key,
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
    fps = 0.0
    n_frames = 0
    t_last = time.perf_counter()
    t_view = t_last
    period = 1.0 / RENDER_HZ
    next_ctrl = time.perf_counter()
    next_render = next_ctrl

    while (not use_mjv or viewer.is_running()) and not quit_req:
        now = time.perf_counter()
        if now >= next_ctrl:
            if cmd.consume_reset():
                reset_to_spawn(model, data, b, settle=True,
                               pos=pos, yaw_deg=yaw_deg)
                print("[RESET] back to spawn", flush=True)
            if hasattr(policy, "select"):
                policy.select(float(cmd.vector()[0]))
            rt.tick()
            next_ctrl = max(next_ctrl + CONTROL_DT, time.perf_counter())
        if now >= next_render:
            cp, cm = ad.camera_poses([CAM])   # rig rides the base
            pipe.set_cam(cp, cm)
            rgb_p, dep_p = pipe.render_frame()
            rgb, depth = pipe.unpack(rgb_p, dep_p)

            if srv is not None:
                rgb_l = rgb[0][:, :, ::-1].copy()
                dep_l = depth_cmap(depth[0])
                bar = np.full((26, rgb_l.shape[1], 3), 24, np.uint8)
                cv2.putText(bar,
                            f"PIE {variant} {fps:4.1f}fps "
                            f"vx={cmd.vector()[0]:+.2f} wz={cmd.vector()[2]:+.2f} "
                            f"cmd={cmd.active():5s} "
                            f"up={rt.base_uprightness():.2f} "
                            f"phy={rt.physics_ms:.1f} pol={rt.policy_ms:.1f}ms "
                            f"xyz={np.round(rt.base_pos(), 1)}",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                bar2 = np.full((24, rgb_l.shape[1], 3), 24, np.uint8)
                cv2.putText(bar2,
                            "GLOBAL arrows: hold Up/Down vx +0.5/-0.2  "
                            "Left/Right wz +/-0.3  ·  release: 1s zero-flush "
                            "then idle",
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
            if web_scene is not None and now - t_view >= period:
                web_scene.update_from_mjdata(data)
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
                print(f"{fps:4.1f}fps  cmd={cmd.active():5s} "
                      f"exp={expert_tag():6s} "
                      f"vx={cmd.vector()[0]:+.2f} wz={cmd.vector()[2]:+.2f} "
                      f"up={rt.base_uprightness():.2f} "
                      f"xyz={np.round(rt.base_pos(), 1)}{cam_txt}", flush=True)
            next_render += period

        wake = min(next_ctrl, next_render) - time.perf_counter()
        if wake > 0:
            time.sleep(wake)

    cmd.stop()
    cam.close()
    if srv is not None:
        srv.close()
    print("bye", flush=True)
    os._exit(0)   # mixed GL atexit teardown segfaults on this driver


if __name__ == "__main__":
    sys.exit(main())
