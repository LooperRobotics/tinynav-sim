#!/usr/bin/env python3
"""Headless 3DGS sensor server: MotrixSim physics + D435i-like rig + IMU.

Self-contained port of the bring-up server (originally in
gs_playground/demo/navigation/sensor_server.py (which itself borrowed helpers
from the upstream robot_locomotion demo). Everything the server needs is
vendored here so the headless path never imports the windowed demo
(motrixsim.render / RenderApp) and never breaks when upstream reshuffles its
demo folder. Heavy assets (scene/robot gaussian plys, MJCF scenes, the
locomotion policy onnx) stay in the gs_playground checkout and are resolved
via GS_PLAYGROUND_ROOT.

Renders without any window:

    infra1 / infra2 : 544x480 L8   (grayscale, 51 mm stereo baseline)
    color           : 544x480 RGB8
    imu             : gyro + accelerometer + orientation @200 Hz (specific force)
    gt              : base pose @50 Hz (ring v2 ground-truth channel)

and writes them into the shared-memory ring consumed by
gsplat/ros/gs_ros_bridge.py, which republishes them as /camera/camera/...
ROS 2 topics plus /sim/gt_pose. D435i intrinsics: fx = fy = 272, cx = 272,
cy = 240 for 544x480 -- the square-pixel/centred model batch_render builds
from fovy, so we pass fovy = 2*atan(240/272) = 82.85 deg and get pixel-exact K.

Usage
-----
    # just check the sensors (dump a few PNG, no ring)
    python sensor_server.py --out /tmp/gsdump --duration 5

    # run the sensor service (ring in /dev/shm, real-time factor 1)
    python sensor_server.py                 # until killed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, HERE.as_posix())
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("ALSOFT_DRIVERS", "null")
GSPG_ROOT_HINT = os.environ.get(
    "GS_PLAYGROUND_ROOT",
    "/workspace/github/simulation/gs_playground"
    if Path("/workspace/github/simulation/gs_playground/demo/navigation").is_dir()
    else "/home/dm/workspace/github/simulation/gs_playground")
_asound = Path(GSPG_ROOT_HINT, "demo/navigation/configs/asound-null.conf")
if _asound.is_file():
    os.environ.setdefault("ALSA_CONFIG_PATH", _asound.as_posix())

from scipy.spatial.transform import Rotation  # noqa: E402

import torch  # noqa: E402
from motrixsim import Body, SceneData, forward_kinematic, msd, step as mx_step  # noqa: E402

from gaussian_renderer import GSRendererMotrixSim  # noqa: E402
from gaussian_renderer.core.util_gau import load_ply  # noqa: E402
from gaussian_renderer.core.batch_rasterization import batch_render  # noqa: E402

from gs_sensor_ring import SensorRingWriter, DEFAULT_PATH, DEFAULT_CMD_FILE  # noqa: E402
from gs_state_shm import StateShmWriter, DEFAULT_PATH as DEFAULT_STATE_PATH  # noqa: E402

CAM_NAMES = ("infra1", "infra2", "color")
W, H, FY = 544, 480, 272.0
FOVY = float(np.degrees(2.0 * np.arctan((H / 2.0) / FY)))     # 82.85 deg = D435i
LUMA = np.array([0.299, 0.587, 0.114], np.float32)

GT_HZ = 50.0          # ground-truth pose cadence (matches the policy control rate)


def gspg_root() -> Path:
    """gs_playground checkout: env override, else container then host layout."""
    env = os.environ.get("GS_PLAYGROUND_ROOT")
    if env:
        return Path(env)
    for cand in ("/workspace/github/simulation/gs_playground",
                 "/home/dm/workspace/github/simulation/gs_playground"):
        if Path(cand, "demo", "navigation").is_dir():
            return Path(cand)
    raise SystemExit("gs_playground checkout not found; set GS_PLAYGROUND_ROOT")


GSPG_NAV = gspg_root() / "demo" / "navigation"


def resolve_gs_path(path: str | Path) -> Path:
    """Config-relative path -> absolute (absolute passes through)."""
    path = Path(path)
    return path if path.is_absolute() else GSPG_NAV / path


# --------------------------------------------------------------------------- #
# vendored from gs_playground demo/navigation (bring-up era)
# --------------------------------------------------------------------------- #
ROBOT_GS_LINK_ALIASES = {
    "go2": {
        "FL_foot": "FL_calf",
        "FR_foot": "FR_calf",
        "RL_foot": "RL_calf",
        "RR_foot": "RR_calf",
    },
}


def collect_scene_gaussians(scene_gaussian_cfg: dict[str, str | Path]) -> dict[str, str]:
    gaussians = {}
    for gs_name, rel_path in scene_gaussian_cfg.items():
        ply = resolve_gs_path(rel_path)
        if ply.exists():
            gaussians[gs_name] = ply.as_posix()
    return gaussians


def collect_robot_gaussians(robot_name: str, model, robot_gs_dir: Path):
    robot_gs_dir = resolve_gs_path(robot_gs_dir)   # config paths are GSPG-relative
    if not robot_gs_dir.exists():
        return {}, {}
    link_names = set(model.link_names)
    aliases = ROBOT_GS_LINK_ALIASES.get(robot_name, {})
    gaussians = {}
    gaussian_links = {}
    skipped = []
    for ply in sorted(robot_gs_dir.glob("*.ply")):
        gs_name = ply.stem
        link_name = gs_name if gs_name in link_names else aliases.get(gs_name)
        if link_name in link_names:
            gaussians[gs_name] = ply.as_posix()
            gaussian_links[gs_name] = link_name
        else:
            skipped.append(gs_name)
    if skipped:
        print(f"Skipping GS assets without matching links: {', '.join(skipped)}", flush=True)
    return gaussians, gaussian_links


def apply_gaussian_links(gs_renderer, model, gaussian_links: dict[str, str]) -> None:
    if not gaussian_links:
        return
    link_ids = {name: idx for idx, name in enumerate(model.link_names)}
    objects_info = []
    body_ids = []
    for gs_name, link_name in gaussian_links.items():
        if gs_name not in gs_renderer.gaussian_start_indices:
            continue
        if link_name not in link_ids:
            continue
        objects_info.append((gs_name, gs_renderer.gaussian_start_indices[gs_name],
                             gs_renderer.gaussian_end_indices[gs_name]))
        body_ids.append(link_ids[link_name])
    if not objects_info:
        return
    gs_renderer.gs_idx_start = np.array([start for _, start, _ in objects_info])
    gs_renderer.gs_idx_end = np.array([end for _, _, end in objects_info])
    gs_renderer.gs_body_ids = np.array(body_ids)
    gs_renderer.set_objects_mapping(objects_info)


class RobotBase:
    def __init__(self, body: Body):
        self._body = body
        self._model = body.model
        self._base_link = body.base_link

    @property
    def num_actuators(self) -> int:
        return self._body.num_actuators

    def dof_pos(self, data: SceneData) -> np.ndarray:
        return self._body.get_joint_dof_pos(data)

    def dof_vel(self, data: SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)

    def base_pose(self, data: SceneData) -> np.ndarray:
        return self._base_link.get_pose(data)

    def set_actuator_ctrls(self, data: SceneData, ctrls: np.ndarray) -> None:
        self._body.set_actuator_ctrls(data, ctrls)

    def gravity(self, data: SceneData) -> np.ndarray:
        rot = self._body.get_rotation_mat(data)
        return rot.T @ np.array([0.0, 0.0, -1.0])


class Go2Robot(RobotBase):
    base_link_name = "base"

    def local_linear_vel(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("local_linvel", data)

    def gyro(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("gyro", data)


class Go2LocomotionPolicy:
    """RL walking policy (onnx, CPU) -- cmd (vx, vy, wz) -> joint targets."""

    _DEFAULT_ANGLES = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])

    def __init__(self, robot, action_scale=0.5, lin_vel_scale=1.0, ang_vel_scale=1.5):
        import onnxruntime as ort
        self._robot = robot
        self.default_angles = self._DEFAULT_ANGLES.copy()
        self.action_scale = action_scale
        self.lin_vel_scale = lin_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.last_action = np.zeros_like(self.default_angles, dtype=np.float32)
        onnx_path = GSPG_NAV / "policies" / "go2_policy.onnx"
        self._policy_session = ort.InferenceSession(onnx_path.as_posix(),
                                                    providers=["CPUExecutionProvider"])
        self._input_name = self._policy_session.get_inputs()[0].name
        self._output_name = self._policy_session.get_outputs()[0].name

    def get_observation(self, data, command):
        lin_vel = self._robot.local_linear_vel(data)
        gyro = self._robot.gyro(data)
        gravity = self._robot.gravity(data)
        dof_pos = self._robot.dof_pos(data)
        dof_vel = self._robot.dof_vel(data)
        obs = np.hstack([lin_vel, gyro, gravity, dof_pos - self.default_angles,
                         dof_vel, self.last_action, command])
        return obs.astype(np.float32)

    def step(self, data, command):
        command = command * np.array([self.lin_vel_scale, self.lin_vel_scale,
                                      self.ang_vel_scale])
        obs = self.get_observation(data, command)
        action = self._policy_session.run(
            [self._output_name], {self._input_name: obs.reshape(1, -1)})[0][0]
        ctrl = action * self.action_scale + self.default_angles
        self._robot.set_actuator_ctrls(data, ctrl)
        self.last_action = action.copy()
        pose = self._robot.base_pose(data)
        z_axis = Rotation.from_quat(pose[3:7]).apply(np.array([0.0, 0.0, 1.0]))
        return float(np.dot(z_axis, np.array([0.0, 0.0, 1.0])) < 0.3)   # is_fallen


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def render_cameras(gs, model, data, cam_ids):
    """One batch_render call for all cameras -> uint8 (N,H,W,3) on CPU."""
    poses = [np.asarray(model.cameras[c].get_pose(data)).ravel() for c in cam_ids]
    cam_pos = np.array([p[:3] for p in poses], np.float32)
    cam_xmat = np.array([Rotation.from_quat(p[3:7]).as_matrix().flatten() for p in poses], np.float32)
    fovy = np.full(len(cam_ids), FOVY, np.float32)
    with torch.no_grad():
        rgb, _depth = batch_render(gs.gaussians, cam_pos, cam_xmat, H, W, fovy)
        u8 = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)      # convert on GPU (3x less PCIe)
        return u8.cpu().numpy()


def self_cull(gs, self_slice, cam_pos, radius, state):
    """Zero the opacity of the robot's own gaussians within `radius` of the camera.

    The per-link splats are fatter than the real limbs, so the snout camera would
    otherwise be buried inside its own torso/leg geometry (a big out-of-focus
    blob over the lower half of the frame).  Restores the previous frame's values
    first, so nothing accumulates.
    """
    g = gs.gaussians
    if state.get("idx") is not None:
        g.opacity[state["idx"]] = state["saved"]
        state["idx"] = None
    if radius <= 0 or self_slice.stop <= self_slice.start:
        return 0
    pts = g.xyz[self_slice]
    d = torch.linalg.norm(pts - torch.as_tensor(cam_pos, device=pts.device).view(1, 3), dim=1)
    near = torch.nonzero(d < radius).flatten()
    if near.numel():
        near = near + self_slice.start
        state["saved"] = g.opacity[near].clone()
        state["idx"] = near
        g.opacity[near] = 0.0
    return int(near.numel())


def to_luma(rgb_u8: np.ndarray) -> np.ndarray:
    return (rgb_u8.astype(np.float32) @ LUMA).astype(np.uint8)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        # bare scene name ("church") resolves to configs/<name>_go2.json
        if path.suffix != ".json":
            path = Path(f"{path}_go2.json")
        path = HERE.parent / "configs" / path
    with path.open() as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="church_go2.json",
                    help="scene config name in gsplat/configs or an absolute path")
    ap.add_argument("--rig-file", default="",
                    help="sensor rig MJCF; default <GSPG>/models/robots/navigation/"
                         "go2/go2_sensor_rig.xml (generate with make_sensor_rig.py)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="simulated seconds to run; 0 = until killed")
    ap.add_argument("--rtf", type=float, default=1.0,
                    help="real-time factor: 1 = real time, 0 = as fast as possible")
    ap.add_argument("--cam-hz", type=float, default=15.0)
    ap.add_argument("--imu-hz", type=float, default=200.0)
    ap.add_argument("--ring", default=DEFAULT_PATH)
    ap.add_argument("--no-ring", action="store_true", help="do not write the shared ring")
    ap.add_argument("--out", default=None, help="dump PNG here and exit (debug)")
    ap.add_argument("--dump-interval", type=float, default=0.5)
    ap.add_argument("--dump-after", type=float, default=1.5)
    ap.add_argument("--cmd-file", default=DEFAULT_CMD_FILE,
                    help="text file with 'vx vy wz' (written by the ROS bridge from /cmd_vel)")
    ap.add_argument("--drive-distance", type=float, default=0.0,
                    help="scripted task: drive straight this many metres along the spawn heading "
                         "(ground-truth heading hold), then stop; 0 = cmd_vel only")
    ap.add_argument("--drive-speed", type=float, default=0.5)
    ap.add_argument("--drive-start", type=float, default=0.0,
                    help="wait this many simulated seconds before starting the drive task")
    ap.add_argument("--self-cull", type=float, default=0.30,
                    help="hide the robot's OWN gaussians within this radius (m) of the onboard "
                         "cameras (0 = disable)")
    ap.add_argument("--window", action="store_true",
                    help="embedded MotrixSim viewer (upstream demo style): physics-mesh "
                         "viewport + left panels for the onboard camera and the 3DGS "
                         "panorama (orbit view). Same session, not a second sim; "
                         "ESC/window close shuts the server down")
    ap.add_argument("--window-fps", type=float, default=60.0,
                    help="viewer frame cap in --window mode (physics stays wall-synced)")
    ap.add_argument("--panel-hz", type=float, default=15.0,
                    help="3DGS panel refresh rate in --window mode; the two panel "
                         "renders share the GPU with the camera stream")
    ap.add_argument("--state-shm", default=DEFAULT_STATE_PATH,
                    help="robot-state slot for the decoupled viewer "
                         "(gsplat/tools/view_window.py); '' disables")
    ap.add_argument("--window-panels", choices=("both", "off"), default="both",
                    help="'off' drops BOTH left image panels (3DGS renders + their "
                         "texture upload/compositing in the viewer); the collision "
                         "viewport stays — less viewer-side CPU per frame")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scene_file = resolve_gs_path(cfg["scene"])
    # The rig must sit next to go2_mjx.xml (MotrixSim resolves its relative
    # mesh paths against the file's own directory), so it is generated INTO
    # the gs_playground checkout by gsplat/robots/go2/make_sensor_rig.py.
    rig_file = Path(args.rig_file) if args.rig_file else (
        GSPG_NAV / "models" / "robots" / "navigation" / "go2" / "go2_sensor_rig.xml")
    if not rig_file.exists():
        raise SystemExit(f"sensor rig missing: {rig_file}\n"
                         f"generate it: /opt/venv_gs/bin/python gsplat/robots/go2/make_sensor_rig.py")
    scene_cfg = {k: Path(v) for k, v in cfg.get("scene_gaussians", {}).items()}
    robot_gs_dir = Path(cfg.get("robot_gs_dir", "assets/go2"))

    print(f"gs_playground: {gspg_root()}")
    print(f"scene : {scene_file}")
    print(f"rig   : {rig_file}")
    scene = msd.from_file(scene_file.as_posix())
    robot_world = msd.from_file(rig_file.as_posix())
    scene.attach(robot_world)
    model = scene.build()

    cam_ids = {}
    for name in CAM_NAMES:
        hit = [i for i, c in enumerate(model.cameras) if c.name == name]
        if not hit:
            raise SystemExit(f"camera {name!r} not found in the rig (got "
                             f"{[c.name for c in model.cameras]})")
        cam_ids[name] = hit[0]
    print("cameras:", {k: v for k, v in cam_ids.items()}, f"fovy={FOVY:.2f} deg "
          f"-> fx=fy={FY:.0f} cx={W/2:.0f} cy={H/2:.0f}")

    data = SceneData(model)
    q0 = cfg.get("initial_qpos")
    if q0:
        qpos = np.asarray(q0, dtype=np.float32)
        if qpos.shape[0] != data.dof_pos.shape[0]:
            raise SystemExit(f"initial_qpos has {qpos.shape[0]} values, "
                             f"expected {data.dof_pos.shape[0]}")
        data.set_dof_pos(qpos, model)
        data.set_dof_vel(np.zeros_like(data.dof_vel))
        forward_kinematic(model, data)          # poses are stale until FK runs
        print(f"initial_qpos applied ({qpos.shape[0]} dof)")
    body = model.get_body("base")
    robot = Go2Robot(body)
    policy = Go2LocomotionPolicy(robot=robot)

    gaussians = collect_scene_gaussians(scene_cfg)
    robot_gaussians, robot_links = collect_robot_gaussians("go2", model, robot_gs_dir)
    gaussians.update(robot_gaussians)
    print(f"gaussians: {len(gaussians)} models ({len(robot_gaussians)} robot links + "
          f"{len(scene_cfg)} scene)")
    gs = GSRendererMotrixSim(gaussians, model)
    apply_gaussian_links(gs, model, robot_links)
    # The scene models come first in the merged buffer (dict insertion order), so
    # everything from n_scene on belongs to the robot itself.  Onboard cameras
    # need those hidden when they are very close (self-occlusion cull).
    n_scene = sum(len(load_ply(resolve_gs_path(scene_cfg[k]).as_posix())) for k in scene_cfg)
    self_slice = slice(n_scene, len(gs.gaussians))
    print(f"3DGS renderer ready. (scene {n_scene:,} + robot "
          f"{len(gs.gaussians)-n_scene:,} gaussians; self-cull r={args.self_cull} m)")

    cull_state: dict = {}
    bp0 = np.asarray(model.get_link("base").get_pose(data)).ravel()
    yaw0 = float(Rotation.from_quat(bp0[3:7]).as_euler("xyz")[2])
    drive = {"active": args.drive_distance > 0, "p0": bp0[:3].copy(), "yaw0": yaw0,
             "dir": np.array([np.cos(yaw0), np.sin(yaw0), 0.0]),
             "wait": args.drive_start}
    if drive["active"]:
        print(f"task: drive {args.drive_distance:.1f} m along heading "
              f"{np.degrees(yaw0):.1f} deg at {args.drive_speed} m/s "
              f"(starting at t={args.drive_start:.1f}s)", flush=True)
    ring = None if args.no_ring else SensorRingWriter(args.ring, w=W, h=H)
    if ring:
        print(f"ring  : {args.ring} ({os.path.getsize(args.ring)/1e6:.1f} MB)")
    # Robot state for the decoupled viewer (keeps its llvmpipe rendering off
    # this process's main thread) — written alongside the GT channel below.
    state_w = None
    if args.state_shm:
        state_w = StateShmWriter(args.state_shm, n_dof=int(np.asarray(data.dof_pos).size))
        print(f"state : {args.state_shm} ({state_w.n_dof} dof)")

    dt = float(model.options.timestep)
    n_ctrl = max(1, round(0.02 / dt))
    cam_period, imu_period, gt_period = 1.0 / args.cam_hz, 1.0 / args.imu_hz, 1.0 / GT_HZ
    next_cam, next_imu, next_gt, next_state = 0.0, 0.0, 0.0, 0.0
    sim_t = 0.0
    step_i = 0
    n_cam = n_imu = n_gt = 0
    n_culled = 0
    t_cull = 0.0
    render_ms = 0.0
    dumps = []
    cmd = np.zeros(3, np.float32)
    t_wall0 = time.monotonic()
    t_last_report = t_wall0

    # ------------------------------------------------------------------ #
    # main loop: the per-step body lives in one closure so the headless
    # path and the RenderApp path (below) run the IDENTICAL body — the
    # windowed viewer shows the live session itself, never a second sim.
    # ------------------------------------------------------------------ #
    def phys_step() -> bool:
        nonlocal sim_t, step_i, n_cam, n_imu, n_gt, n_culled, t_cull, render_ms
        nonlocal next_cam, next_imu, next_gt, next_state, cmd, t_last_report
        mx_step(model, data)
        sim_t += dt
        step_i += 1

        if step_i % n_ctrl == 0:
            if args.cmd_file:
                try:
                    parsed = np.loadtxt(args.cmd_file, dtype=np.float32).reshape(-1)
                    if parsed.size >= 3:      # empty/torn read (atomic-replace
                        cmd = parsed[:3]      # race): keep the previous command
                except Exception:
                    pass
            if drive["active"] and drive["wait"] > 0 and sim_t < drive["wait"]:
                cmd = np.zeros(3, np.float32)
            elif drive["active"]:                   # scripted straight-line task
                bp = np.asarray(model.get_link("base").get_pose(data)).ravel()
                yaw = float(Rotation.from_quat(bp[3:7]).as_euler("xyz")[2])
                err = (drive["yaw0"] - yaw + np.pi) % (2 * np.pi) - np.pi
                done = float(np.dot(bp[:3] - drive["p0"], drive["dir"]))
                if done >= args.drive_distance:
                    drive["active"] = False
                    cmd = np.zeros(3, np.float32)
                    print(f"task done: travelled {done:.2f} m", flush=True)
                else:
                    cmd = np.array([args.drive_speed, 0.0, np.clip(1.5 * err, -1.0, 1.0)],
                                   np.float32)
            need_reset = policy.step(data, cmd)
            if need_reset:
                data.reset(model)

        # ---- IMU (accumulator: the physics step does not divide 200 Hz) ----
        if sim_t >= next_imu:
            gyro = np.asarray(model.get_sensor_value("imu_gyro", data)).ravel()
            accel = np.asarray(model.get_sensor_value("imu_accel", data)).ravel()
            quat = np.asarray(model.get_sensor_value("imu_quat", data)).ravel()
            if ring:
                ring.publish_imu(sim_t, gyro, accel, quat)
            n_imu += 1
            next_imu += imu_period

        # ---- ground truth (base pose in the MJCF world frame) ----
        if ring and sim_t >= next_gt:
            bp = np.asarray(model.get_link("base").get_pose(data)).ravel()
            ring.publish_gt(sim_t, bp[:3], bp[3:7])
            n_gt += 1
            next_gt += gt_period

        # ---- viewer state (same cadence; dof_pos holds the base pose too) ----
        if state_w is not None and sim_t >= next_state:
            state_w.publish(sim_t, np.asarray(data.dof_pos).ravel())
            next_state += gt_period

        # ---- cameras ----
        if sim_t >= next_cam:
            gs.update_gaussians(data)
            t0 = time.perf_counter()
            cam_pos0 = np.asarray(model.cameras[cam_ids["color"]].get_pose(data)).ravel()[:3]
            n_culled = self_cull(gs, self_slice, cam_pos0, args.self_cull, cull_state)
            t_cull = time.perf_counter() - t0
            t0 = time.perf_counter()
            frames = render_cameras(gs, model, data, [cam_ids[n] for n in CAM_NAMES])
            render_ms = (time.perf_counter() - t0) * 1e3
            infra1 = to_luma(frames[0])
            infra2 = to_luma(frames[1])
            color = frames[2]
            if args.window:
                from PIL import Image
                last_color[0] = np.asarray(Image.fromarray(color).resize((480, 360)))
            if ring:
                ring.publish_cameras(infra1, infra2, color, sim_t)
            n_cam += 1
            if args.out and sim_t >= args.dump_after and len(dumps) < 3 and \
                    (not dumps or sim_t - dumps[-1][0] > args.dump_interval):
                dumps.append((sim_t, infra1.copy(), infra2.copy(), color.copy(), render_ms))
            next_cam += cam_period

        if time.monotonic() - t_last_report > 5.0:
            t_last_report = time.monotonic()
            print(f"  t={sim_t:6.2f}s  cam={n_cam} imu={n_imu} gt={n_gt}  "
                  f"wall={time.monotonic()-t_wall0:6.1f}s  render={render_ms:5.1f} ms  "
                  f"self-cull={n_culled} ({t_cull*1e3:.1f} ms)", flush=True)
        return args.duration <= 0 or sim_t < args.duration

    print(f"running ({'until killed' if args.duration <= 0 else f'{args.duration:.1f} s sim'}, "
          f"{args.cam_hz:.0f} Hz cameras, {args.imu_hz:.0f} Hz IMU, {GT_HZ:.0f} Hz GT, "
          f"rtf={args.rtf}, window={args.window})...")

    if args.window:
        # Upstream-style viewer embedded in THIS session: main viewport =
        # physics meshes (collision groups on), left panels = onboard camera
        # + 3DGS panorama following the orbit camera. render_loop paces
        # physics to wall time (rtf=1 semantics; rtf=0 drops the frame cap).
        # NB run.render_loop IGNORES render_step's return value — ESC/window
        # close must raise to actually leave the loop and reach the cleanup.
        #
        # The rig's /root/.bashrc exports the NVIDIA PRIME/EGL trio for rviz
        # and gz (the container has no NVIDIA EGL that wgpu can use: Mesa
        # llvmpipe drives the viewport fine, while NVIDIA EGL makes wgpu's
        # khronos-egl probe panic with unwrap() on None). Docker-exec and
        # launch paths differ in whether they inherit it — drop it HERE so
        # every launcher gets a working window.
        for _v in ("__GLX_VENDOR_LIBRARY_NAME", "__NV_PRIME_RENDER_OFFLOAD",
                   "__EGL_VENDOR_LIBRARY_FILENAMES"):
            os.environ.pop(_v, None)
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
        from motrixsim.render import Layout, RenderApp
        from motrixsim import run as mx_run

        prof = {"phys": 0.0, "panel": 0.0, "sync": 0.0, "t0": time.monotonic()}

        def profiled_phys():
            t0 = time.perf_counter()
            ok = phys_step()
            prof["phys"] += time.perf_counter() - t0
            return ok

        # Panel budget: the camera batch already costs ~58 ms x 15 Hz (~0.87 of
        # the GPU second, near the headless ceiling), so the viewer must be
        # nearly free or rtf collapses. Mitigations: the onboard-camera panel
        # REUSES the camera stream's own frame (no extra render, and the true
        # 82.85 deg fovy -- gs.render would show the MJCF-ignored 45 deg), and
        # only the panorama panel renders, rate-limited to --panel-hz.
        # --window-panels off drops both widgets entirely.
        panels_on = args.window_panels != "off"
        panel_stride = max(1, round(args.window_fps / max(args.panel_hz, 1.0)))
        tick = [0]
        last_color = [None]

        def render_step():
            tick[0] += 1
            if panels_on and tick[0] % panel_stride == 0:
                if last_color[0] is not None:
                    head_img.pixels = last_color[0]
                # NB: no gs.update_gaussians(data) here — the camera tick in
                # phys_step already refreshed the link transforms, and a
                # second robot-gaussian upload per panel frame costs real
                # GPU time. The panorama lags one camera tick (<=67 ms).
                t0 = time.perf_counter()
                results = gs.render(model, data, [-1], 480, 360,
                                    system_camera=render.system_camera)
                if -1 in results:
                    rgb = results[-1][0].cpu().numpy()
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                    pan_img.pixels = rgb
                prof["panel"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            render.sync(data)
            prof["sync"] += time.perf_counter() - t0
            if time.monotonic() - prof["t0"] > 5.0:
                prof["t0"] = time.monotonic()
                wall5 = 5.0
                print(f"  [win prof] phys {1000*prof['phys']/wall5:6.0f} ms/s  "
                      f"panel {1000*prof['panel']/wall5:6.0f} ms/s  "
                      f"sync {1000*prof['sync']/wall5:6.0f} ms/s", flush=True)
                prof["phys"] = prof["panel"] = prof["sync"] = 0.0
            for key in ("esc", "escape"):
                if render.input.is_key_just_pressed(key):
                    raise KeyboardInterrupt("ESC pressed — shutting the sensor server down")

        with RenderApp() as render:
            render.launch(model)
            render.opt.set_group_vis(2, True)
            render.opt.set_group_vis(3, True)
            if panels_on:
                head_img = render.create_image(np.zeros((360, 480, 3), np.uint8))
                render.widgets.create_image_widget(
                    head_img, layout=Layout(left=10, top=10, width=480, height=360))
                pan_img = render.create_image(np.zeros((360, 480, 3), np.uint8))
                render.widgets.create_image_widget(
                    pan_img, layout=Layout(left=10, top=370, width=480, height=360))
            print(f"windowed (panels={args.window_panels}): main viewport = "
                  "physics meshes (orbit/zoom with the mouse); ESC = quit server",
                  flush=True)
            # Own loop instead of mx_run.render_loop: that helper updates its
            # physics timebase AFTER the step loop, so each frame credits only
            # the render time to the accumulator — the step-loop wall time is
            # dropped and the sim settles at rtf ~0.5 no matter the hardware
            # (measured: 0.45-0.49 across panel/fps configurations). Here the
            # accumulating interval covers the whole frame.
            frame_dt = 1.0 / args.window_fps if args.rtf > 0 else 0.0
            if args.rtf == 0:
                frame_dt = 0.0            # spin: physics as fast as possible
            phys_remain = 0.0
            t_prev = time.monotonic()
            try:
                while True:
                    t0 = time.monotonic()
                    phys_remain = min(phys_remain + (t0 - t_prev), 0.25)  # cap: no spiral
                    t_prev = t0
                    while phys_remain > dt:
                        if not profiled_phys():
                            raise KeyboardInterrupt("duration reached")
                        phys_remain -= dt
                    render_step()
                    if frame_dt > 0.0:
                        slack = frame_dt - (time.monotonic() - t0)
                        if slack > 0:
                            time.sleep(slack)
            except KeyboardInterrupt as exc:
                print(f"window closed/ESC ({exc}) — exiting", flush=True)
    else:
        while phys_step():
            # ---- pacing ----
            if args.rtf > 0:
                target = t_wall0 + sim_t / args.rtf
                lag = target - time.monotonic()
                if lag > 0:
                    time.sleep(lag)

    wall = time.monotonic() - t_wall0
    print(f"done: {sim_t:.2f} s sim in {wall:.2f} s wall (rtf {sim_t/wall:.2f}); "
          f"cameras {n_cam} ({n_cam/max(sim_t,1e-9):.1f} Hz), IMU {n_imu} "
          f"({n_imu/max(sim_t,1e-9):.1f} Hz)", flush=True)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for i, (t, i1, i2, c, ms) in enumerate(dumps):
            Image.fromarray(i1).save(out / f"infra1_{i}.png")
            Image.fromarray(i2).save(out / f"infra2_{i}.png")
            Image.fromarray(c).save(out / f"color_{i}.png")
            print(f"  dumped t={t:.2f}s (render {ms:.1f} ms) -> {out}")
    if ring:
        ring.close(unlink=False)   # keep the ring for the bridge to read
    if state_w is not None:
        # unlink on exit: a stale slot would make a late viewer show a frozen
        # robot; the file reappears with the next server.
        state_w.close(unlink=True)
    print("served.")


if __name__ == "__main__":
    main()
