#!/usr/bin/env python3
"""HIL host: strategy-driven MuJoCo dog + contract sensor face on DDS
(rclpy node /insight_full, same domain as the tinynav stack -- the mjsim
container is the operative host; see docker/mujoco.Dockerfile and the HIL
section in ../README.md).

Contract (hil/__init__.py, aligned with tinynav-sim/docs/looper-contract.md
plus stereo + imu extensions): infra1/infra2 mono8, depth mono16mm, color
JPEG, vio_image — ExactTime same stamp per frame (default 10 Hz); vio_100hz
100 Hz; vio_status 1 Hz latched; tf_static re-published every 5 s; imu
~200 Hz BEST_EFFORT. Stamps are the wall time at which the pose/state was
SAMPLED (采集时刻位姿的时间); the physics thread also feeds substep IMU/base
samples so vio_100hz and imu bracket every camera stamp.

depth = wgpu splat depth (route-B precedent: pixel-aligned with infra1,
mono16 mm; the policy's internal 60x106 mujoco depth is untouched). The
camera_info K is read back from the render pipeline after the contract
fovy is applied, so camera_info and the rendered pixels never drift.

Usage:
    python hil.py 5070 [--scene map3|map2] [--spawn <scene's>]
        [--cam-hz 10] [--vio-hz 100]
        [--imu-every 1]
        [--preview [--p-host H --p-port N]] [--no-viewer]
        [--view web|glfw [--web-port 8012]]  (web default: headless viser
        viewer, ~0.04 core @10Hz vs ~0.69 for glfw; no DISPLAY needed)

Commands: the stack's /cmd_vel drives; physically held arrow keys override
it (release runs the gazebo 1s zero-flush, then DDS resumes). Reset/Stop
are the web GUI buttons only -- no keyboard bindings for them, on purpose:
a stray R press must not reset a live HIL session.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../mujoco
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import ctypes
    ctypes.CDLL("libc.so.6").prctl(15, b"gausscam-looper", 0, 0, 0)
except Exception:   # noqa: BLE001 -- cosmetic only (comm name)
    pass
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
cv2.setNumThreads(0)
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from sim.plant import (  # noqa: E402
    SCENES,
    build_model,
    get_scene,
    reset_to_spawn,
)
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402
from sim.keyboard import HoldCmd, KeyboardOverride  # noqa: E402

from hil import (  # noqa: E402
    CONTRACT_FOVY_DEG,
    CONTRACT_H,
    CONTRACT_W,
    TelemetrySink,
    mat_to_quat_xyzw,
    optical,
    quat_xyzw_to_mat,
    rigid_inv,
    rigid_mul,
)

CONTROL_DT = 0.02
CAMS = ("infra1", "infra2")       # stereo pair the face renders


def _argv_value(flag: str, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = argv[0] if argv else "5070"
    scene_name = _argv_value("--scene", "map3")
    sc = get_scene(scene_name)
    spawn = _argv_value("--spawn", sc.default_spawn)
    cam_hz = float(_argv_value("--cam-hz", "10.0"))
    view_hz = float(_argv_value("--view-hz", "10.0"))   # viewer.sync rate: CPU scales linearly (render-per-sync)
    vio_hz = float(_argv_value("--vio-hz", "100.0"))
    imu_every = int(_argv_value("--imu-every", "1"))
    preview = "--preview" in sys.argv
    use_viewer = "--no-viewer" not in sys.argv
    view_mode = _argv_value("--view", "web")      # web | glfw
    web_port = int(_argv_value("--web-port", "8012"))
    wall_alpha = float(_argv_value("--wall-alpha", "0.1"))
    if view_mode == "web":
        use_viewer = False    # web viewer replaces the GLFW window entirely
    p_host = _argv_value("--p-host", "127.0.0.1")
    p_port = int(_argv_value("--p-port", "8888"))
    if spawn not in sc.spawns:
        raise SystemExit(f"unknown spawn {spawn!r}; choose {sorted(sc.spawns)}")
    pos, yaw_deg = sc.spawns[spawn]

    model, b = build_model(scene_name)
    model.vis.quality.shadowsize = 0
    data = mujoco.MjData(model)
    reset_to_spawn(model, data, b, settle=True, pos=pos, yaw_deg=yaw_deg)

    # passive viewer (default ON): observation only, 12s wedge guard.
    # NO key_callback on purpose: Reset/Stop are the web GUI buttons only,
    # so a stray R keypress can never reset a live HIL session.
    viewer = None
    keys = {"reset": False, "stop": False}

    # wgpu splat pipeline BEFORE any GLX/EGL context: on the 2060M's NVIDIA
    # 595 driver, a live GLX viewer context at wgpu adapter-enumeration time
    # panics the khronos-egl unwrap inside wgpuInstanceEnumerateAdapters
    # (reproduced in-container; view.py/view_pie already follow
    # this wgpu-first order, hil.py was written on the 5070 where order is
    # free).
    links = sorted(["base_link"] + [leg + part
                                    for leg in ("FL", "FR", "RL", "RR")
                                    for part in ("_hip", "_thigh", "_calf")])
    from gausscam.adapters.mujoco import MuJoCoAdapter
    from gausscam.backends.webgpu import Pipeline, pick_variant
    from gausscam.core.assets import GaussianCloud
    ad = MuJoCoAdapter(model, data)
    cpos, cxm = ad.camera_poses(list(CAMS))
    # Static scene straight from the 3DGS PLY (gausscam >= 0.1.1), dog block
    # dropped: the contract cameras sit at the head front and never see the
    # body, so robot gaussians pay projection time for nothing (view.py's
    # infra1 preview rig reached the same verdict and renders static-only
    # too). from_ply returns RAW INRIA values --
    # activate (exp/sigmoid) and slice to DC before the pipeline; empty
    # slots = static-only cloud (scene_n == N, update_links copies through).
    _scene_ply = _HERE.parent / "model" / "splat" / "map3_scene.ply"
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
    }
    d["W"] = np.int32(CONTRACT_W)
    d["H"] = np.int32(CONTRACT_H)
    d["fovy"] = np.float32(CONTRACT_FOVY_DEG)
    d["cam_pos"] = cpos.astype(np.float32)
    d["cam_xmat"] = cxm.astype(np.float32)
    variant = pick_variant(sel)
    pipe = Pipeline(d, sel, variant=variant)
    fxy, cx, cy = pipe._K
    print(f"contract face: {CONTRACT_W}x{CONTRACT_H} "
          f"fovy={CONTRACT_FOVY_DEG:.3f} -> K fx=fy={fxy:.4f} "
          f"cx={cx:.1f} cy={cy:.1f}", flush=True)
    print(f"device: {pipe.adapter.info['device']}  variant={variant}", flush=True)

    # policy depth camera next: wgpu -> EGL(depth) -> EGL(viewer). Empirical
    # in-container order on the 2060M/595 driver — every other permutation
    # breaks (GLX before wgpu panics enumeration; viewer-EGL before the
    # depth EGL fails its makeCurrent).
    policy_cam = DepthCamera(model, b.depth_camera_id)

    web_scene = None
    if view_mode == "web":
        # headless web viewer (viser + patched mjviser): no GLFW/DISPLAY
        # dependency, ~0.04 core at 10 Hz vs ~0.69 for the GLFW viewer.
        # update_from_mjdata reads the LIVE mjData -> control thread only,
        # throttled at view_hz (never the render thread -- same race family
        # as mj_copyData).
        try:
            import viser
            from webviewer import ViserMujocoScene
            # shaft walls: BLEND alpha stacks PER-PANEL now (webviewer
            # emits translucent geoms as separate meshes), so the XML's
            # 0.1 alpha renders as real see-through glass. Stairs, landings
            # and the dog stay opaque so the structure reads.
            walls = ((model.geom_bodyid == 0)
                     & (model.geom_rgba[:, 3] < 0.5))
            model.geom_rgba[walls, 3] = wall_alpha
            web_server = viser.ViserServer(port=web_port)
            web_scene = ViserMujocoScene(web_server, model, num_envs=1)
            # "Track camera" force-aims every client at the world ORIGIN
            # (look_at=zeros) -- the dog lives at the spawn, not the origin,
            # so tracking off + spawn framing below.
            web_scene.camera_tracking_enabled = False
            web_scene.create_visualization_gui()

            def _frame_spawn(client):
                def _apply():
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
                # late re-apply: the client re-frames itself once the scene
                # upload lands, so a single set at connect gets overwritten.
                _apply()
                for d in (1.5, 3.0, 6.0):
                    threading.Timer(d, _apply).start()

            web_server.on_client_connect(_frame_spawn)

            # Reset/Stop live ONLY here in hil (no keyboard bindings):
            # the buttons set the flags the control loop consumes.
            def _add_cmd_button(label: str, key: str) -> None:
                btn = web_server.gui.add_button(label)

                @btn.on_click
                def _(_event) -> None:
                    keys[key] = True

            _add_cmd_button("Reset to spawn", "reset")
            _add_cmd_button("Stop", "stop")
            web_scene.show_contact_points = True
            # hulls render at 0.5 opacity over EVERY geom (walls, dog) and
            # read as opaque -- collision viewing is covered by the G3 group
            # toggle instead.
            web_scene.show_convex_hull = False
            web_scene.update_from_mjdata(data)
            # viser silently bumps to the next port when the requested one
            # is busy -- print the ACTUAL bound port, not the request.
            print(f"[WEBVIEWER] http://0.0.0.0:{web_server.get_port()}  "
                  f"(wall alpha={wall_alpha}, contact points on, "
                  "GUI: Reset/Stop)", flush=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[WEBVIEWER] init failed: {exc}", flush=True)
            web_scene = None

    if use_viewer:
        holder: dict = {}
        ready = threading.Event()

        def _open_viewer():
            try:
                import glfw
                # No GLX in this process: GLX viewer + wgpu(Vulkan) + EGL
                # cannot coexist on the 2060M/595 driver (wgpu enumerate
                # panics if GLX comes first; EGL makeCurrent fails once GLX
                # and wgpu are both up — both reproduced in-container
                # ). GLFW hints are global state of the shared
                # instance and the C++ window creation goes through the same
                # handle, so this hint switches the viewer to EGL too.
                glfw.window_hint(glfw.CONTEXT_CREATION_API,
                                 glfw.EGL_CONTEXT_API)
                holder["v"] = mujoco.viewer.launch_passive(model, data)
            except Exception as exc:      # noqa: BLE001
                print(f"[VIEWER] launch failed: {exc}", flush=True)
            finally:
                ready.set()

        threading.Thread(target=_open_viewer, daemon=True).start()
        if not ready.wait(timeout=12.0):
            print("[VIEWER] window not mapped in 12s (compositor wedged?) "
                  "-- continuing headless", flush=True)
        else:
            viewer = holder.get("v")
            if viewer is not None:
                # free camera: tracking (follow base_link) locked lookat, so
                # right-drag pan was dead -- left as free for orbit/pan/zoom.
                print("[VIEWER] mapped (free camera; Reset/Stop are the "
                      "web GUI buttons)",
                      flush=True)


    # camera mount T_base_camera (left optical), measured once at spawn
    root = b.root_qpos_adr
    T_wb = (quat_xyzw_to_mat(data.qpos[root + 3:root + 7]),
            np.asarray(data.qpos[root:root + 3], float))
    cp, cm = ad.camera_poses(list(CAMS))
    T_wc = optical((np.asarray(cm[0], float), np.asarray(cp[0], float)))
    t_base_cam = rigid_mul(rigid_inv(T_wb), T_wc)
    print(f"mount T_base_camera: t={np.round(t_base_cam[1], 4).tolist()}",
          flush=True)

    sink = TelemetrySink(
        gyro_adr=b.gyro_adr, acc_adr=b.acc_adr,
        root_qpos_adr=b.root_qpos_adr, t_base_cam=t_base_cam)

    from hil.ros import RosFace
    face = RosFace(sink, vio_hz, imu_every)

    # command priority: physically held arrows override the stack's
    # /cmd_vel (human nudge during e2e runs); release runs the gazebo
    # 1s zero-flush, then the DDS source flows again.
    kb = HoldCmd()
    cmd_mux = KeyboardOverride(kb, face.cmd)

    policy = PieOnnxPolicy(_HERE / "assets" / "policy" / sc.hil_policy,
                           provider="cpu")
    rt = PieRuntime(model, b, data, policy, policy_cam, cmd_mux,
                    spawn_pos=pos, spawn_yaw_deg=yaw_deg,
                    substep_hook=sink.sample)

    srv = None
    if preview:
        from view import MjpegServer
        srv = MjpegServer(p_host, p_port)
        print(f"preview: http://{p_host}:{p_port}/ "
              "(scene view + dog-eye)", flush=True)

    n_frames = 0
    n_encode_fail = 0
    render_ms = 0.0
    fps = 0.0
    t_last = time.perf_counter()
    n_last = 0
    t_view = t_last
    t_tf = 0.0
    t_status = 0.0
    period = 1.0 / cam_hz
    next_ctrl = time.perf_counter()

    # Pose snapshot slot: the control thread writes after every tick (refs
    # only -- each tick allocates fresh arrays, nothing is mutated in place),
    # the render thread reads. Rendering therefore NEVER blocks physics/IMU:
    # a slow GPU just lowers the frame rate (README cross-GPU table) while
    # the 200 Hz control loop and the substep IMU keep full cadence. THE
    # stamp is the pose sample time taken here, exactly the contract.
    snap_lock = threading.Lock()
    snap: dict = {"stamp": 0.0, "pf": None, "qf": None, "cp": None,
                  "cm": None, "base": (0.0, 0.0, 0.0), "up": 1.0,
                  "qpos": None}
    render_failed = 0

    def render_loop() -> None:
        """Frame producer at cam_hz: snapshot -> wgpu render -> encode ->
        publish. Owns ALL frame-set work (incl. JPEG + DDS publish,
        both thread-safe) so the control thread only writes the slot."""
        nonlocal n_frames, n_encode_fail, render_ms, fps
        nonlocal t_tf, t_status, render_failed
        scene_renderer = None
        scene_cam = None
        data_copy = None
        if preview:
            from view import depth_cmap
            # offscreen EGL context lives in THIS thread (EGL is per-thread).
            # The scene view renders a thread-PRIVATE MjData restored from
            # the snapshot qpos -- mj_copyData against the live mjData races
            # mj_step ("stack is in use" FatalError); model is read-only and
            # per-thread MjData is the documented-safe pattern.
            scene_renderer = mujoco.Renderer(model, height=480, width=640)
            scene_cam = mujoco.MjvCamera()
            scene_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            scene_cam.trackbodyid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            scene_cam.distance = 2.0
            scene_cam.elevation = -25
            scene_cam.azimuth = 90
            data_copy = mujoco.MjData(model)
        next_frame = time.perf_counter()
        f_count, f_win = 0, time.perf_counter()
        while running and (viewer is None or viewer.is_running()):
            now = time.perf_counter()
            if now < next_frame:
                time.sleep(min(next_frame - now, 0.02))
                continue
            next_frame = max(next_frame + period, now)
            try:
                with snap_lock:
                    stamp = snap["stamp"]
                    pf, qf = snap["pf"], snap["qf"]
                    cp, cm = snap["cp"], snap["cm"]
                    base, up = snap["base"], snap["up"]
                    qpos = snap["qpos"]
                if pf is None:
                    continue
                t0 = time.perf_counter()
                pipe.set_links(pf, qf)
                pipe.set_cam(cp, cm)
                rgb_p, dep_p = pipe.render_frame()
                rgb, depth = pipe.unpack(rgb_p, dep_p)

                gray_l = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY)
                gray_r = cv2.cvtColor(rgb[1], cv2.COLOR_RGB2GRAY)
                ok, jpg = cv2.imencode(".jpg", rgb[0][:, :, ::-1],
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    n_encode_fail += 1
                    jpg = None

                R_opt, t_opt = optical((np.asarray(cm[0], float),
                                        np.asarray(cp[0], float)))
                face.publish_frame(stamp, gray_l, gray_r, depth[0],
                                   mat_to_quat_xyzw(R_opt), t_opt,
                                   jpg.tobytes() if jpg is not None else None)
                face.publish_camera_info(stamp, fxy, cx, cy)

                wall = time.time()
                if wall - t_tf >= 5.0:
                    face.publish_tf(wall)
                    t_tf = wall
                if wall - t_status >= 1.0:
                    face.publish_status(wall)
                    t_status = wall

                if srv is not None:
                    dep_l = depth_cmap(depth[0])
                    bar = np.full((26, rgb[0].shape[1], 3), 24, np.uint8)
                    cv2.putText(bar,
                                f"HIL {variant} {fps:4.1f}fps "
                                f"xyz={np.round(base, 1)} up={up:.2f}",
                                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    dog_eye = cv2.vconcat([bar, rgb[0][:, :, ::-1], dep_l])
                    data_copy.qpos[:] = qpos
                    mujoco.mj_forward(model, data_copy)
                    scene_renderer.update_scene(data_copy, camera=scene_cam)
                    scene = scene_renderer.render()[:, :, ::-1].copy()
                    h = dog_eye.shape[0]
                    w = int(scene.shape[1] * h / scene.shape[0])
                    scene = cv2.resize(scene, (w, h),
                                       interpolation=cv2.INTER_AREA)
                    img = cv2.hconcat([dog_eye, scene])
                    srv.push(cv2.imencode(
                        ".jpg", img,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())

                render_ms = (time.perf_counter() - t0) * 1e3
                n_frames += 1
                f_count += 1
                now2 = time.perf_counter()
                if now2 - f_win >= 1.0:
                    fps = f_count / (now2 - f_win)
                    f_count, f_win = 0, now2
            except Exception:                 # noqa: BLE001
                render_failed += 1
                if render_failed == 1 or render_failed % 50 == 0:
                    import traceback
                    traceback.print_exc()
                next_frame = time.perf_counter() + 1.0

    print(f"HIL face up: transport=ros  cam {cam_hz:.0f}Hz  "
          f"vio {vio_hz:.0f}Hz  imu every {imu_every} substep(s)", flush=True)
    running = True
    render_thread = threading.Thread(target=render_loop, daemon=True)
    render_thread.start()
    failed = False
    try:
        while running and (viewer is None or viewer.is_running()):
            if keys["reset"]:
                keys["reset"] = False
                reset_to_spawn(model, data, b, settle=True,
                               pos=pos, yaw_deg=yaw_deg)
                print("[RESET] back to spawn", flush=True)
            if keys["stop"]:
                keys["stop"] = False
                cmd_mux.zero()
                print("[STOP] command zeroed (GUI Stop)", flush=True)
            now = time.perf_counter()
            if now >= next_ctrl:
                rt.tick()
                next_ctrl = max(next_ctrl + CONTROL_DT, time.perf_counter())
                # pose snapshot for the render thread: THE stamp = pose
                # sample time. Refs only -- each tick allocates fresh
                # arrays, so the render thread can hold them lock-free.
                pl = ad.body_poses(links)
                cp, cm = ad.camera_poses(list(CAMS))    # rides the base
                with snap_lock:
                    snap["stamp"] = time.time()
                    snap["pf"] = np.stack(
                        [np.asarray(pl[n][0], np.float32) for n in links])
                    snap["qf"] = np.stack(
                        [np.asarray(pl[n][1], np.float32) for n in links])
                    snap["cp"] = np.asarray(cp, np.float32)
                    snap["cm"] = np.asarray(cm, np.float32)
                    snap["base"] = tuple(np.round(rt.base_pos(), 3))
                    snap["up"] = float(rt.base_uprightness())
                    snap["qpos"] = rt.data.qpos.copy()
                if not np.all(np.isfinite(rt.data.qpos)):
                    running = False

            now2 = time.perf_counter()
            # launch_passive renders the viewer-side mjData copy; without a
            # periodic sync() the window shows a static launch-time frame
            # forever (the original "container viewer freeze"). view_hz caps
            # the scene-repaint cost.
            if web_scene is not None and now2 - t_view >= 1 / view_hz:
                web_scene.update_from_mjdata(data)
                t_view = now2
            elif viewer is not None and now2 - t_view >= 1 / view_hz:
                viewer.sync()
                t_view = now2
            if now2 - t_last >= 1.0:
                fps = (n_frames - n_last) / (now2 - t_last)
                n_last = n_frames
                t_last = now2
                print(f"cam {fps:4.1f}Hz render {render_ms:5.1f}ms "
                      f"kb {cmd_mux.state:<5} "
                      f"imu {face.n_imu} vio {face.n_vio} "
                      f"dropped {sink.dropped} clip {face.cmd.clipped} "
                      f"encfail {n_encode_fail}"
                      + (f" RFAIL {render_failed}" if render_failed else ""),
                      flush=True)
                face.n_imu = face.n_vio = 0
            wake = next_ctrl - time.perf_counter()
            if wake > 0:
                time.sleep(wake)
    except BaseException:
        import traceback
        traceback.print_exc()
        failed = True
    finally:
        print("closing...", flush=True)
        running = False
        render_thread.join(timeout=2.0)
        policy_cam.close()
        face.close()
        if srv is not None:
            srv.close()
        print("bye", flush=True)
        os._exit(1 if failed else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
