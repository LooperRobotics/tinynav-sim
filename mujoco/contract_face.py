#!/usr/bin/env python3
"""The looper perception face as its own process, spawned by dog.py --contract.

The split is load-bearing: this image's unitree SDK binding and the rmw link
cyclonedds into one process, and cyclonedds 0.10.x allows a domain id to be
created exactly once per process -- the SDK's ChannelFactoryInitialize(0) and
rclpy's node cannot coexist here. dog.py stays pure-SDK; this file stays
pure-rclpy (no unitree_sdk2py import; mujoco only for the mesh/native render
arms); the two meet on the wire like any two containers, with plant state
handed over on a /dev/shm ring.

What it does = hil.py's render loop, minus the body: reads the state ring,
composes each contract camera's world pose from the base pose via the mounts
dog.py measured at spawn, renders the stereo pair, and publishes the looper
contract as rclpy node `insight_full`.  Two render arms:

- `splat` (map3/map2): gausscam RGB pair; depth selectable with --depth:
  `mesh` (mujoco collision-world truth, pixel-aligned with the contract K),
  `stereo` (retinify on the rendered pair), `splat` (gausscam depth).
- `mujoco` (floorplan): native MuJoCo Renderer at the contract stereo pair,
  RGB + left depth from sim/render_contract.py; no PLY.

    python3 contract_face.py --ply model/splat/map3_scene.ply \
        --mounts '<json from dog.py>' [--cam-hz 10] [--vio-hz 100] \
        [--imu-every 1] [--gpu 5070]
    python3 contract_face.py --scene floorplan --mounts '<json>'   # native
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from hil import (  # noqa: E402
    BASELINE,
    CONTRACT_FOVY_DEG,
    CONTRACT_H,
    CONTRACT_W,
    TelemetrySink,
    mat_to_quat_xyzw,
    optical,
    quat_wxyz_to_mat,
    rigid_mul,
)

SHM_PATH = "/dev/shm/mjsim_emu_state.bin"
# must match dog.py.EmuStateRing
_REC = struct.Struct("<Q d 3d 4d 3d 3d")
_REC_SIZE = _REC.size
CAMS = ("infra1", "infra2")


class StateRing:
    """Reader side of dog.py's ring (see EmuStateRing there for the layout).
    MAP_SHARED read-only mapping of the same /dev/shm file."""

    def __init__(self, path: str) -> None:
        import mmap
        for _ in range(100):               # dog creates the file before spawn
            if os.path.exists(path):
                break
            time.sleep(0.05)
        fd = os.open(path, os.O_RDWR)
        try:
            self._buf = mmap.mmap(fd, _REC_SIZE, mmap.MAP_SHARED,
                                  mmap.PROT_READ)
        finally:
            os.close(fd)

    def read(self):
        seq1 = self._buf[0:8]
        raw = self._buf[:]
        seq2 = self._buf[0:8]
        if seq1 != seq2 or seq1 == b"\0" * 8:
            return None
        vals = _REC.unpack(raw)
        return vals[1], vals[2:5], vals[5:9], vals[9:12], vals[12:15]


class _ShimData:
    """Duck-typed MjData for TelemetrySink.sample: .sensordata = [gyro, acc],
    .qpos = [pos, quat_wxyz] with the addresses fixed at construction."""

    def __init__(self, gyro, acc, pos, quat) -> None:
        self.sensordata = np.concatenate([gyro, acc]).astype(np.float64)
        self.qpos = np.concatenate([pos, quat]).astype(np.float64)


def build_pipeline(ply: Path, gpu: str, cam_pos, cam_xmat):
    from gausscam.backends.webgpu import Pipeline, pick_variant
    from gausscam.core.assets import GaussianCloud

    cloud = GaussianCloud.from_ply(ply)
    d = {
        "xyz": cloud.xyz,
        "rot": cloud.rot,
        "scale": np.exp(cloud.scale).astype(np.float32),
        "opacity": (1.0 / (1.0 + np.exp(-cloud.opacity))).astype(np.float32),
        "sh": np.ascontiguousarray(cloud.sh[:, :3]),
        # static-scene cloud: the contract cameras sit at the head and never
        # see the body (hil.py's dog-block verdict), so no link slots.
        "slots": np.zeros(0, np.int32),
    }
    d["W"] = np.int32(CONTRACT_W)
    d["H"] = np.int32(CONTRACT_H)
    d["fovy"] = np.float32(CONTRACT_FOVY_DEG)
    # the ctor inverts the view matrices, so these must be REAL poses: the
    # first ring record composed through the mounts (set_cam refreshes them
    # every frame afterwards).
    d["cam_pos"] = cam_pos
    d["cam_xmat"] = cam_xmat
    variant = pick_variant(gpu)
    pipe = Pipeline(d, gpu, variant=variant)
    return pipe, variant


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", default=None,
                    help="3DGS ply (splat renderer only; optional for scenes "
                         "with render='mujoco')")
    ap.add_argument("--mounts", required=True,
                    help="JSON: [{raw:(R,t), optical:(R,t)}, ...] per cam")
    ap.add_argument("--cam-hz", type=float, default=10.0)
    ap.add_argument("--vio-hz", type=float, default=100.0)
    ap.add_argument("--imu-every", type=int, default=1)
    ap.add_argument("--gpu", default="5070")
    ap.add_argument("--render", default="auto",
                    choices=["auto", "splat", "mujoco"],
                    help="auto = the scene's render field; splat = gausscam "
                         "RGB+depth; mujoco = native Renderer stereo/depth")
    ap.add_argument("--floorplan-config", default=None,
                    help="floorplan YAML (procedural scene only)")
    ap.add_argument("--depth", default="mesh",
                    choices=["mesh", "stereo", "splat"],
                    help="splat scenes only: mesh = mujoco-rendered "
                         "collision-world truth; stereo = retinify net on the "
                         "rendered pair; splat = gausscam depth.  The native "
                         "mujoco renderer always emits its own depth")
    ap.add_argument("--scene", default="map3",
                    help="scene whose model the native renderer / mesh depth "
                         "builds")
    ap.add_argument("--shm", default=SHM_PATH)
    args = ap.parse_args()

    from sim.plant import build_model, get_scene

    sc = get_scene(args.scene)
    render_mode = args.render
    if render_mode == "auto":
        render_mode = "mujoco" if sc.render == "mujoco" else "splat"
    if render_mode == "splat" and not args.ply:
        raise SystemExit("--ply is required for the splat renderer")

    mounts = json.loads(args.mounts)
    # JSON turns tuples into lists; rigid math wants arrays.
    mount_raw = [(np.asarray(m["raw"][0], float), np.asarray(m["raw"][1], float))
                 for m in mounts]
    mount_opt = [(np.asarray(m["optical"][0], float),
                  np.asarray(m["optical"][1], float)) for m in mounts]

    if render_mode == "splat":
        # Render the right eye at the CONTRACT baseline, not the emu model's.
        # The go2 model puts the infra pair ~5.1 cm apart along the image
        # axis, while the real camera (and everything downstream:
        # camera_info's P, the disparity->depth conversion) declares
        # hil.BASELINE.  The shift is along the LEFT camera's raw-frame x
        # (image right); a base-frame shift would slide the eye along the
        # view direction and move nothing.
        ax = mount_raw[0][0] @ np.array([1.0, 0.0, 0.0])  # left image-x
        b_emu = float((mount_raw[1][1] - mount_raw[0][1]) @ ax)
        t_r = mount_raw[0][1] + BASELINE * ax
        print(f"stereo: emu baseline {b_emu * 100:.2f} cm -> right eye "
              f"rendered at {BASELINE * 100:.2f} cm (contract)", flush=True)
        mount_raw[1] = (mount_raw[1][0], t_r)

    ring = StateRing(args.shm)

    def _cam_poses(rec):
        _t, pos, quat, _gyro, _acc = rec
        # The ring carries MuJoCo's WXYZ verbatim (dog.py packs qpos as-is) --
        # feeding it to the XYZW parser rotated every camera pose by a large
        # constant and skewed every render after the first frame.
        T_wb = (quat_wxyz_to_mat(quat), np.asarray(pos, float))
        cp = np.stack([rigid_mul(T_wb, mount_raw[i])[1]
                       for i in range(len(CAMS))]).astype(np.float32)
        cm = np.stack([rigid_mul(T_wb, mount_raw[i])[0]
                       for i in range(len(CAMS))]).astype(np.float32)
        return cp, cm

    pipe = None
    variant = render_mode
    mujoco_model = None
    left_cam = right_cam = None
    mesh_depth = None
    stereo = None

    if render_mode == "mujoco":
        from sim.floorplan import load_params

        params = (load_params(args.floorplan_config)
                  if args.floorplan_config else None)
        mujoco_model, _b = build_model(args.scene, params)
        left_cam = _b.contract_left_camera_id
        right_cam = _b.contract_right_camera_id
        fovy = float(mujoco_model.cam_fovy[left_cam])
        fxy = 0.5 * CONTRACT_H / np.tan(np.radians(fovy) / 2.0)
        cx, cy = CONTRACT_W / 2.0, CONTRACT_H / 2.0
        print(f"render: mujoco native stereo/depth via "
              f"infra1_contract/infra2_contract (fovy {fovy:.3f})",
              flush=True)
        print("depth: mujoco renderer left camera", flush=True)
    else:
        rec0 = None
        for _ in range(200):                # dog ticks from the moment it spawns
            rec0 = ring.read()
            if rec0 is not None:
                break
            time.sleep(0.05)
        if rec0 is None:
            raise SystemExit(
                "no plant state on the ring after 10s -- is dog.py up?")
        cam_pos, cam_xmat = _cam_poses(rec0)

        pipe, variant = build_pipeline(Path(args.ply), args.gpu, cam_pos,
                                       cam_xmat)
        fxy, cx, cy = pipe._K

        # GL-context ordering is load-bearing: this runs alongside the gausscam
        # wgpu pipeline (wgpu-first is the proven order -- see hil.py's note).
        # The EGL context is built lazily on FIRST CALL, not here: EGL contexts
        # are thread-affine, mesh_depth runs on the render-loop thread, and a
        # context built on this (main) thread fails every eglMakeCurrent there
        # with EGL_BAD_ACCESS.
        if args.depth == "mesh":
            _mesh: dict = {}

            def mesh_depth(pos_w, quat_w):
                """One contract mesh-depth frame (uint16 mm, 0 = no hit)."""
                if not _mesh:
                    import mujoco as _mj

                    model, _b = build_model(args.scene)
                    cam_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_CAMERA,
                                            "infra1_contract")
                    if cam_id < 0:
                        raise SystemExit(
                            "infra1_contract camera missing -- the scene model "
                            "predates plant.CONTRACT_DEPTH_CAMERA")
                    # World-only mask: every static geom (catch floor + scene
                    # boxes) lives in group 0 and every robot geom in groups
                    # 2/3, so masking to group 0 removes the dog's own shell
                    # without touching the model.
                    rend = _mj.Renderer(model, height=CONTRACT_H,
                                        width=CONTRACT_W)
                    rend.enable_depth_rendering()
                    rend.scene.flags[_mj.mjtRndFlag.mjRND_SHADOW] = 0
                    rend.scene.flags[_mj.mjtRndFlag.mjRND_REFLECTION] = 0
                    opt = _mj.MjvOption()
                    opt.geomgroup[:] = 0
                    opt.geomgroup[0] = 1   # the static world, nothing else
                    far = float(model.vis.map.zfar * model.stat.extent)
                    _mesh.update(mj=_mj, model=model, data=_mj.MjData(model),
                                 rend=rend, opt=opt, cam=cam_id, far=far)
                    print(f"depth: mujoco mesh via infra1_contract "
                          f"(contract fovy, far {far:.1f} m)", flush=True)
                m = _mesh
                m["data"].qpos[0:3] = pos_w
                m["data"].qpos[3:7] = quat_w   # ring WXYZ, MuJoCo's own
                m["mj"].mj_forward(m["model"], m["data"])
                m["rend"].update_scene(m["data"], camera=m["cam"],
                                       scene_option=m["opt"])
                d = np.asarray(m["rend"].render(), dtype=np.float32)
                d[(~np.isfinite(d)) | (d <= 0.0)
                  | (d >= m["far"] * 0.99)] = 0.0
                return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

        if args.depth == "stereo":
            from sim.stereo import RetinifyStereo

            stereo = RetinifyStereo()
            print(f"depth: retinify net @ {BASELINE * 100:.2f} cm baseline",
                  flush=True)
        elif args.depth == "splat":
            print("depth: splat render (comparison arm)", flush=True)
        print(f"device: {pipe.adapter.info['device']}  variant={variant}",
              flush=True)

    print(f"contract face: {CONTRACT_W}x{CONTRACT_H} "
          f"fovy={CONTRACT_FOVY_DEG:.3f} -> K fx=fy={fxy:.4f} "
          f"cx={cx:.1f} cy={cy:.1f}", flush=True)

    sink = TelemetrySink(gyro_adr=0, acc_adr=3, root_qpos_adr=0,
                         t_base_cam=mount_opt[0])

    from hil.ros import RosFace
    face = RosFace(sink, args.vio_hz, args.imu_every)

    running = True

    def _stop(_sig, _frm) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # feeder: ring -> sink at imu cadence (the sink dedups nothing; the face's
    # fast-pub thread drains it for imu + vio_100hz).
    def feeder() -> None:
        last_t = 0.0
        while running:
            rec = ring.read()
            if rec is not None and rec[0] != last_t:
                last_t = rec[0]
                t, pos, quat, gyro, acc = rec
                sink.sample(_ShimData(np.asarray(gyro), np.asarray(acc),
                                      np.asarray(pos), np.asarray(quat)))
            else:
                time.sleep(0.001)

    threading.Thread(target=feeder, daemon=True, name="ring-feeder").start()

    # dummy link slots: the cloud carries none, set_links writes nothing --
    # kept for pipeline-call parity with hil.py's render loop.
    dummy_pf = np.zeros((len(CAMS) * 6 + 1, 3), np.float32)
    dummy_qf = np.zeros((len(CAMS) * 6 + 1, 4), np.float32)

    import cv2
    cv2.setNumThreads(0)

    def render_loop() -> None:
        period = 1.0 / args.cam_hz
        next_frame = time.perf_counter()
        n_frames, render_ms, failed_n = 0, 0.0, 0
        f_count, f_win, cam_fps = 0, time.perf_counter(), 0.0
        mujoco_rend = None
        if mujoco_model is not None:
            # EGL contexts are thread-affine: create the renderer HERE, not on
            # the main thread.
            from sim.render_contract import MujocoContractRenderer

            mujoco_rend = MujocoContractRenderer(
                mujoco_model, left_cam, right_cam)
            print(f"[face] mujoco renderer K={mujoco_rend.K}", flush=True)
        while running:
            now = time.perf_counter()
            if now < next_frame:
                time.sleep(min(next_frame - now, 0.02))
                continue
            next_frame = max(next_frame + period, now)
            try:
                rec = ring.read()
                if rec is None:
                    continue
                stamp, pos, quat, _gyro, _acc = rec
                t0 = time.perf_counter()
                if mujoco_rend is not None:
                    (gray_l, gray_r, depth_l, R_opt, t_opt,
                     jpg) = mujoco_rend.render(pos, quat)
                    jpg_len = len(jpg) if jpg is not None else 0
                else:
                    cp, cm = _cam_poses(rec)
                    pipe.set_links(dummy_pf, dummy_qf)
                    pipe.set_cam(cp, cm)
                    rgb_p, dep_p = pipe.render_frame()
                    rgb, depth = pipe.unpack(rgb_p, dep_p)

                    gray_l = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY)
                    gray_r = cv2.cvtColor(rgb[1], cv2.COLOR_RGB2GRAY)
                    if mesh_depth is not None:
                        # Collision-world truth, 10 Hz with the render loop;
                        # the wire encoding (uint16 mm, 0 = invalid) matches
                        # the other arms.
                        depth_l = mesh_depth(pos, quat)
                    elif stereo is not None:
                        _disp, d_m = stereo.infer(gray_l, gray_r, BASELINE,
                                                  fxy)
                        depth_l = np.clip(d_m * 1000.0, 0, 65535).astype(
                            np.uint16)
                    else:
                        depth_l = depth[0]
                    ok, jpg_buf = cv2.imencode(
                        ".jpg", rgb[0][:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                    jpg = jpg_buf.tobytes() if ok else None
                    jpg_len = len(jpg) if jpg is not None else 0
                    R_opt, t_opt = optical((cm[0], cp[0]))
                if n_frames == 0:
                    print(f"[face] payload bytes: depth={depth_l.nbytes} "
                          f"gray={gray_l.nbytes} jpg={jpg_len} "
                          f"subs_depth={face.pub['depth'].get_subscription_count()} "
                          f"subs_vio={face.pub['vio_100hz'].get_subscription_count()}",
                          flush=True)
                face.publish_frame(
                    stamp, gray_l, gray_r, depth_l,
                    mat_to_quat_xyzw(R_opt), t_opt, jpg)
                face.publish_camera_info(stamp, fxy, cx, cy)

                wall = time.time()
                if not hasattr(render_loop, "_t_tf"):
                    render_loop._t_tf = 0.0
                    render_loop._t_status = 0.0
                if wall - render_loop._t_tf >= 5.0:
                    face.publish_tf(wall)
                    render_loop._t_tf = wall
                if wall - render_loop._t_status >= 1.0:
                    face.publish_status(wall)
                    render_loop._t_status = wall

                render_ms = (time.perf_counter() - t0) * 1e3
                n_frames += 1
                f_count += 1
                now2 = time.perf_counter()
                if now2 - f_win >= 1.0:
                    cam_fps = f_count / (now2 - f_win)
                    f_count, f_win = 0, now2
                    print(f"face cam {cam_fps:4.1f}Hz render {render_ms:5.1f}ms "
                          f"imu {face.n_imu} vio {face.n_vio} "
                          f"dropped {sink.dropped} "
                          f"subs_d={face.pub['depth'].get_subscription_count()}",
                          flush=True)
                    face.n_imu = face.n_vio = 0
            except Exception:              # noqa: BLE001
                failed_n += 1
                if failed_n == 1 or failed_n % 50 == 0:
                    import traceback
                    traceback.print_exc()
                next_frame = time.perf_counter() + 1.0

    print(f"contract face up: cam {args.cam_hz:.0f}Hz  vio {args.vio_hz:.0f}Hz  "
          f"imu every {args.imu_every} substep(s)", flush=True)
    threading.Thread(target=render_loop, daemon=True,
                     name="render").start()

    while running:
        time.sleep(0.2)
    print("contract face closing...", flush=True)
    face.close()
    print("contract face bye", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
