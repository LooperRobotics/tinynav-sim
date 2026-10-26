#!/usr/bin/env python3
"""robot_emu host: the MuJoCo go2 body wearing a real robot's Unitree-DDS
face (docs/multi-robot-sim-testbench.md §5). Production pilot code connects
unmodified over loopback; the emu answers sport RPCs with per-model FSM
gating and streams state topics at a bench-friendly rate (20 Hz default).

    python3 dog.py --profile as2 --iface lo [--scene map2] [--state-hz 20]
        [--view web [--web-port 8013]]
        [--contract]

--contract additionally serves the looper perception face -- as a SEPARATE
process (contract_face.py, spawned here) fed plant state over a /dev/shm
ring. The split is not a style choice: this image's unitree SDK binding and
the rmw link the same libddsc, and cyclonedds 0.10.x allows a domain id to be
created exactly once per process -- the SDK's ChannelFactoryInitialize(0) and
rclpy's node therefore cannot coexist in one process at all (the fleet
chassis gets away with it because its rmw is Fast DDS -- an entirely
different middleware). dog.py stays a pure-SDK process, contract_face.py a
pure-rclpy process, and the two meet on the wire like any two containers.

Command authority: sport Move RPCs (the pilot chassis's real drive path).
With --contract, physically held arrow keys override the RPC stream via the
gazebo release semantics (sim.keyboard.KeyboardOverride) -- human nudge
during e2e runs, never the other way round.

hil.py stays the perception-only host (a /cmd_vel-driven body without the
chassis face); this one is the full-stack emu.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../mujoco
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from robot_emu import fsm as F  # noqa: E402
from robot_emu.cmd import EmuCmdSource  # noqa: E402
from robot_emu.profile import get_profile  # noqa: E402
from robot_emu.scenarios import ScenarioService, ScenarioState  # noqa: E402
from robot_emu.services import SportService  # noqa: E402
from robot_emu.state_feed import Snapshot, StateFeed  # noqa: E402

CONTROL_DT = 0.02
CAMS = ("infra1", "infra2")       # stereo pair the contract face renders
SHM_PATH = "/dev/shm/mjsim_emu_state.bin"
# seq(u64) | t(f64) | pos(3f64) | quat_wxyz(4f64) | gyro(3f64) | acc(3f64)
_REC = struct.Struct("<Q d 3d 4d 3d 3d")
_REC_SIZE = _REC.size


class EmuStateRing:
    """Single-writer / single-reader handoff to contract_face.py. The writer
    packs a full record per substep (imu cadence) with a monotonic seq; the
    reader retries on a torn read instead of locking -- one cache line class
    of struct, 200 Hz, no reason to pay a mutex for it. FILE-backed mmap:
    anonymous mmap is private per process on Linux, so the ring lives as a
    /dev/shm file mapped MAP_SHARED by both sides."""

    def __init__(self, path: str = SHM_PATH) -> None:
        import mmap
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, _REC_SIZE)
            self._buf = mmap.mmap(fd, _REC_SIZE)
        finally:
            os.close(fd)
        self._buf.seek(0)
        self._buf.write(b"\0" * _REC_SIZE)
        self._seq = 0

    def write(self, t: float, pos, quat_wxyz, gyro, acc) -> None:
        self._seq += 1
        self._buf.seek(0)
        self._buf.write(_REC.pack(self._seq, t, *pos, *quat_wxyz, *gyro, *acc))

    def read(self):
        """Latest record as (t, pos, quat, gyro, acc), or None before the
        first write / under a torn read (seq changed across our copy)."""
        seq1 = self._buf[0:8]
        raw = self._buf[:]
        seq2 = self._buf[0:8]
        if seq1 != seq2 or seq1 == b"\0" * 8:
            return None
        vals = _REC.unpack(raw)
        return vals[1], vals[2:5], vals[5:9], vals[9:12], vals[12:15]


# Full-qpos side-channel for offboard video capture: the state ring carries
# the base only (all the contract face needs), and a BEV video replayed from
# those base poses would show the legs standing still.  Same single-writer
# discipline, one record per control tick.
QPOS_PATH = "/dev/shm/mjsim_emu_qpos.bin"
QPOS_CAP = 64
_QHDR = struct.Struct("<QdI")


class QposTap:
    def __init__(self, nq: int, path: str = QPOS_PATH) -> None:
        import mmap
        self._nq = min(int(nq), QPOS_CAP)
        self._size = _QHDR.size + 8 * self._nq
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, self._size)
            self._buf = mmap.mmap(fd, self._size)
        finally:
            os.close(fd)
        self._buf.seek(0)
        self._buf.write(b"\0" * self._size)
        self._seq = 0

    def write(self, t: float, qpos) -> None:
        self._seq += 1
        self._buf.seek(0)
        self._buf.write(_QHDR.pack(self._seq, t, self._nq))
        self._buf.write(np.asarray(qpos[:self._nq], dtype=np.float64).tobytes())


def _measure_camera_mounts(model, data, b) -> list[dict]:
    """Per-contract-camera rigid mount T_base_camera (raw mujoco camera
    frame, NOT optical -- contract_face.py applies optical() per publish,
    exactly like hil.py's render loop). Measured once at spawn."""
    from gausscam.adapters.mujoco import MuJoCoAdapter
    from hil import optical, quat_wxyz_to_mat, rigid_inv, rigid_mul

    ad = MuJoCoAdapter(model, data)
    # qpos free joint is WXYZ (MuJoCo order) -- the XYZW parser here put a
    # large constant rotation into the mount's world side.
    T_wb = (quat_wxyz_to_mat(data.qpos[3:7]),
            np.asarray(data.qpos[0:3], float))
    T_wb_inv = rigid_inv(T_wb)
    mounts = []
    cp, cm = ad.camera_poses(list(CAMS))
    for i in range(len(CAMS)):
        T_wc = optical((np.asarray(cm[i], float), np.asarray(cp[i], float)))
        # keep BOTH frames: raw (for pipe.set_cam) and optical (publish path);
        # lists, not ndarrays -- this crosses to the face process as JSON
        raw = rigid_mul(T_wb_inv,
                        (np.asarray(cm[i], float), np.asarray(cp[i], float)))
        opt = rigid_mul(T_wb_inv, T_wc)
        mounts.append({
            "raw": [np.asarray(raw[0]).tolist(), np.asarray(raw[1]).tolist()],
            "optical": [np.asarray(opt[0]).tolist(),
                        np.asarray(opt[1]).tolist()],
        })
    return mounts


def _spawn_contract_face(ply: Path | None, mounts, args) -> subprocess.Popen:
    cmd = [sys.executable, "-u", str(_HERE / "contract_face.py")]
    if ply is not None:
        cmd += ["--ply", str(ply)]
    cmd += ["--mounts", json.dumps(mounts),
            "--scene", args.scene,
            "--cam-hz", str(args.cam_hz),
            "--vio-hz", str(args.vio_hz),
            "--imu-every", str(args.imu_every),
            "--depth", args.depth,
            "--gpu", args.gpu]
    if args.floorplan_config:
        cmd += ["--floorplan-config", args.floorplan_config]
    return subprocess.Popen(cmd)


def _quat_wxyz_to_rpy(q) -> tuple:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class _SnapSlot:
    """Control-thread writes, feed-thread reads; refs only, GIL swaps."""

    def __init__(self) -> None:
        self._snap = None

    def set(self, snap) -> None:
        self._snap = snap

    def get(self):
        return self._snap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True,
                    help="emu model profile (go2/go2w/b2/b2w/as2)")
    ap.add_argument("--iface", default="lo",
                    help="network interface the emu DDS binds (default lo)")
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--spawn", default=None)
    ap.add_argument("--state-hz", type=float, default=20.0)
    ap.add_argument("--view", default="none", choices=["none", "web"])
    ap.add_argument("--web-port", type=int, default=8013)
    ap.add_argument("--contract", action="store_true",
                    help="serve the looper perception face via "
                         "contract_face.py (splat scenes need the ply; "
                         "floorplan uses the native mujoco renderer)")
    ap.add_argument("--floorplan-config", default=None,
                    help="floorplan YAML (procedural floorplan scene)")
    ap.add_argument("--cam-hz", type=float, default=10.0)
    ap.add_argument("--vio-hz", type=float, default=100.0)
    ap.add_argument("--imu-every", type=int, default=1)
    ap.add_argument("--expert", default="auto",
                    choices=["auto", "stairs", "flat", "single"],
                    help="expert selector: auto = backward commands ride the "
                         "flat omni net, the rest the scene's own expert "
                         "(stairs/flat force one side; single pins --policy)")
    ap.add_argument("--policy", default=None,
                    help="ONNX pinned by --expert single "
                         "(default: the scene's expert)")
    ap.add_argument("--depth", default="mesh",
                    choices=["mesh", "stereo", "splat"],
                    help="contract face depth: mesh = mujoco collision-world "
                         "truth; stereo = retinify net on the rendered pair; "
                         "splat = gausscam depth render")
    ap.add_argument("--gpu", default="5070",
                    help="gausscam GPU selector (pick_variant name match)")
    args = ap.parse_args()

    import mujoco  # noqa: E402
    from sim.plant import (active_spawns, build_model, get_scene,
                           reset_to_spawn)  # noqa: E402
    from sim.floorplan import load_params  # noqa: E402
    from sim.policy import PieOnnxPolicy, load_policy  # noqa: E402
    from sim.expert import DualExpertPolicy  # noqa: E402
    from sim.runtime import DepthCamera, PieRuntime  # noqa: E402

    profile = get_profile(args.profile)
    sc = get_scene(args.scene)
    spawn = args.spawn or sc.default_spawn

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, args.iface)

    fsm = F.EmuFSM(profile.fsm_table)
    fsm.set_mode(profile.initial_mode)
    emu_cmd = EmuCmdSource(profile.vel_clip)
    scenarios = ScenarioState()
    scenario_svc = ScenarioService(scenarios)
    sport = SportService(fsm, emu_cmd, scenarios)

    model, b = build_model(
        args.scene,
        load_params(args.floorplan_config) if args.floorplan_config else None)
    spawns = active_spawns()
    if spawn not in spawns:
        raise SystemExit(
            f"unknown spawn {spawn!r}; choose {sorted(spawns)}")
    pos, yaw_deg = spawns[spawn]
    # go2 body invariant: root free joint (6) + 12 leg dofs. The snapshot
    # builder reads qvel[0:3] as the root linear velocity; a joint-order
    # change would otherwise corrupt vel_body silently.
    assert model.nv == 18, f"expected the go2 body (nv=18), got nv={model.nv}"
    model.vis.quality.shadowsize = 0
    data = mujoco.MjData(model)
    reset_to_spawn(model, data, b, settle=True, pos=pos, yaw_deg=yaw_deg)
    qpos_tap = QposTap(model.nq)

    # Dual experts on stairs scenes (sim/expert.py): the scene's own veteran
    # for forward/side commands, the flat omni net for backward. On the flat
    # scene the primary IS the omni net -- a dual pair of the same file would
    # only reset its GRU on every direction flip.
    _flat = "policy_24000_flat_omni.onnx"
    _primary = (Path(args.policy) if args.policy
                else _HERE / "assets" / "policy" / sc.hil_policy)
    policy = load_policy(_primary, provider="cpu")
    if (isinstance(policy, PieOnnxPolicy) and args.expert != "single"
            and sc.hil_policy != _flat):
        policy = DualExpertPolicy(_primary, _HERE / "assets" / "policy" / _flat,
                                  mode=args.expert, provider="cpu")
    policy_cam = DepthCamera(model, b.depth_camera_id)

    keys = {"reset": False, "stop": False}
    web_scene = None
    if args.view == "web":
        try:
            import viser
            from webviewer import ViserMujocoScene
            # shaft walls translucent (hil.py's BLEND-per-panel note)
            walls = ((model.geom_bodyid == 0)
                     & (model.geom_rgba[:, 3] < 0.5))
            model.geom_rgba[walls, 3] = 0.1
            web_server = viser.ViserServer(port=args.web_port)
            web_scene = ViserMujocoScene(web_server, model, num_envs=1)
            web_scene.camera_tracking_enabled = False
            web_scene.create_visualization_gui()

            # Framing via the server's initial_camera: the client applies it
            # itself during init (its built-in default aims at the world
            # ORIGIN -- what a fresh page shows until it lands). The
            # per-client re-apply timers this replaces fired blind for 6 s
            # and yanked the camera whenever the operator zoomed in that
            # window after a refresh.
            p = np.asarray(pos, float)
            web_server.initial_camera.position = tuple(
                p + np.array([-0.35, -1.6, 0.6]))
            web_server.initial_camera.look_at = tuple(p)

            def _add_cmd_button(label: str, key: str) -> None:
                btn = web_server.gui.add_button(label)

                @btn.on_click
                def _(_event) -> None:
                    keys[key] = True

            _add_cmd_button("Reset to spawn", "reset")
            _add_cmd_button("Stop", "stop")
            web_scene.show_contact_points = True
            web_scene.show_convex_hull = False
            web_scene.update_from_mjdata(data)
            # viser silently bumps to the next port when busy -- print the
            # ACTUAL bound port, not the request.
            print(f"[WEBVIEWER] http://0.0.0.0:{web_server.get_port()}",
                  flush=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[WEBVIEWER] init failed: {exc}", flush=True)
            web_scene = None

    face_proc = None
    if args.contract:
        if not sc.ply and sc.render != "mujoco":
            raise SystemExit(
                f"scene {args.scene!r} has no splat ply -- the contract face "
                "needs a splat scene (map3|map2) or a native-render scene "
                "(floorplan)")
        mounts = _measure_camera_mounts(model, data, b)
        ring = EmuStateRing()
        ply = (Path(_HERE.parent / "model" / "splat" / sc.ply)
               if sc.ply else None)
        face_proc = _spawn_contract_face(ply, mounts, args)

        def _substep_hook(mjdata) -> None:
            """Per-substep: feed the contract face's ring (imu cadence)."""
            root = b.root_qpos_adr
            ring.write(
                time.time(),
                mjdata.qpos[root:root + 3],
                mjdata.qpos[root + 3:root + 7],
                mjdata.sensordata[b.gyro_adr:b.gyro_adr + 3],
                mjdata.sensordata[b.acc_adr:b.acc_adr + 3],
            )
        substep_hook = _substep_hook
    else:
        ring = None
        substep_hook = None

    kb = None
    if args.contract:
        from sim.keyboard import HoldCmd, KeyboardOverride
        kb = HoldCmd()
        cmd_source = KeyboardOverride(kb, emu_cmd)
    else:
        cmd_source = emu_cmd

    rt = PieRuntime(model, b, data, policy, policy_cam, cmd_source,
                    spawn_pos=pos, spawn_yaw_deg=yaw_deg,
                    substep_hook=substep_hook)

    snap_slot = _SnapSlot()
    feed = StateFeed(profile, fsm, scenarios, snap_slot.get,
                     hz=args.state_hz)
    feed.start()

    print(f"emu up: profile={profile.name} iface={args.iface} "
          f"scene={args.scene}/{spawn} state_hz={args.state_hz:.0f} "
          f"mode={fsm.mode} contract={face_proc is not None}", flush=True)

    t0 = time.monotonic()
    t_view = 0.0
    t_stat = 0.0
    next_ctrl = time.perf_counter()
    running = True
    failed = False
    try:
        while running:
            if keys["reset"]:
                keys["reset"] = False
                reset_to_spawn(model, data, b, settle=True,
                               pos=pos, yaw_deg=yaw_deg)
                print("[RESET] back to spawn", flush=True)
            if keys["stop"]:
                keys["stop"] = False
                cmd_source.zero()
                print("[STOP] command zeroed (GUI Stop)", flush=True)
            if scenarios.consume_reset():
                reset_to_spawn(model, data, b, settle=True,
                               pos=pos, yaw_deg=yaw_deg)
                print("[RESET] back to spawn", flush=True)
            now = time.perf_counter()
            if now >= next_ctrl:
                sel = getattr(policy, "select", None)
                if sel is not None:
                    sel(float(cmd_source.vector()[0]))
                rt.tick()
                next_ctrl = max(next_ctrl + CONTROL_DT, time.perf_counter())
                qpos_tap.write(time.time(), data.qpos)
                root = b.root_qpos_adr
                quat = tuple(data.qpos[root + 3:root + 7])
                rot = _quat_wxyz_to_mat(quat)
                vel_world = np.asarray(data.qvel[0:3], dtype=np.float64)
                uptime = time.monotonic() - t0
                snap_slot.set(Snapshot(
                    uptime=uptime,
                    pos=tuple(float(v) for v in data.qpos[root:root + 3]),
                    quat_wxyz=quat,
                    rpy=_quat_wxyz_to_rpy(quat),
                    vel_body=tuple(rot.T @ vel_world),
                    yaw_rate=float(data.sensordata[b.gyro_adr + 2]),
                    gyro=tuple(float(v)
                               for v in data.sensordata[b.gyro_adr:b.gyro_adr + 3]),
                    acc=tuple(float(v)
                              for v in data.sensordata[b.acc_adr:b.acc_adr + 3]),
                    joint_q=tuple(float(v)
                                  for v in data.qpos[b.joint_qpos_adr]),
                    joint_dq=tuple(float(v)
                                   for v in data.qvel[b.joint_dof_adr]),
                ))
                if not np.all(np.isfinite(data.qpos)):
                    running = False
            now2 = time.perf_counter()
            if web_scene is not None and now2 - t_view >= 1 / 10.0:
                web_scene.update_from_mjdata(data)
                t_view = now2
            if now2 - t_stat >= 1.0:
                t_stat = now2
                c = cmd_source.vector()
                p = rt.base_pos()
                extra = (f" kb {getattr(cmd_source, 'state', '-')}"
                         if face_proc is not None else "")
                print(f"mode={fsm.mode} sport_req={sport.n_requests} "
                      f"move_ok={emu_cmd.moves_accepted} "
                      f"move_refused={emu_cmd.moves_refused} "
                      f"cmd=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) "
                      f"pub={feed.pub_count} xyz=({p[0]:.2f},{p[1]:.2f},"
                      f"{p[2]:.2f}) up={rt.base_uprightness():.2f}"
                      + extra, flush=True)
            wake = next_ctrl - time.perf_counter()
            if wake > 0:
                time.sleep(wake)
    except BaseException:
        import traceback
        traceback.print_exc()
        failed = True
    finally:
        print("closing...", flush=True)
        if face_proc is not None:
            face_proc.terminate()
            try:
                face_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                face_proc.kill()
        feed.stop()
        policy_cam.close()
        print("bye", flush=True)
        os._exit(1 if failed else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
