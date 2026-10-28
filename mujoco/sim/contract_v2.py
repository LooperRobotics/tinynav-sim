"""Policy contracts of the depth-student family and the blind policies.

unitree_rl_mjlab Unitree-<Robot>-Depth-Student exports, one per robot:
go2_depth_v2 (Go2), as2_depth_v1 / b2_depth_v1 (legged), go2w_depth_v1
(wheel-legged) and the wheel-legged *_depth_v2 (go2w / b2w / as2w: no phase,
low-speed brake). Physics rate, depth camera and depth preprocessing are the
PIE contract's (sim/contract.py, imported, not copied); joints, default
pose, observation terms and action layout come from the ONNX metadata, so
one class serves every robot:

    base_ang_vel(3) projected_gravity(3) command(3) phase(2)
    joint_pos(P) joint_vel(J) actions(A)

in that order, minus the terms the metadata `observation_names` leaves out
(the wheel-legged v2 exports have no phase: 53-D), plus heading_error(1) last
for the heading students (contract `<contract>_heading`, below).

J = metadata joint_names (every joint, model order), P = the position-
controlled joints of action_layout (legs; relative to the default pose --
wheel angles are unbounded and left out), A = the action layout's position
then velocity targets. Legged: P = J = A = 12 (47-D). Wheel-legged: P = 12,
J = A = 16 (55-D). go2_depth_v2 exports predate action_layout: all 12 joints
position-controlled at scale 0.25.

phase = (sin, cos)(2 pi * t / 0.6 s), t = control_step * CONTROL_DT counted
from the last reset, and zero while |command| < 0.1 (training also zeroes
commands up to that norm, so the same deadzone is applied to the command).
Mirrors unitree_rl_mjlab src/tasks/velocity/mdp/observations.py::phase.
Exports with metadata `gait_ref_speed` (go2-blind's legged finals of
2026-10-27 and the students distilled from them) run a speed clock instead
(GaitClock): the period shortens above the reference speed and the phase
advances every control step, as go2-blind's mdp.gait_clock.

Wheel brake (metadata `wheel_brake_rad_s`, wheel-legged v2): with the shaped
command zero and every velocity-controlled joint slower than that, the
velocity targets are zero (unitree_rl_mjlab mdp/wheel_action.py; applied by
the runtime).

Heading students (unitree_rl_mjlab Unitree-<Robot>-Depth-Student-Heading,
contract `<student contract>_heading`, e.g. go2_depth_v2_heading): trained
on go2-blind's straight walking with a heading-error input, so their command
envelope is the blind joystick range. heading_error = the heading the
command asks for minus the base heading (rad, wrapped, + = turn left): the
target heading is the base yaw for the first two control steps after a
reset, then integrates the shaped yaw-rate command every control step, held
within +-0.5 rad of the base yaw (anti-windup when the robot cannot follow).
HeadingError mirrors unitree_rl_mjlab mdp/observations.py::heading_error and
the go2-blind deploy observation of the same name.

BlindContract: unitree_rl_mjlab Unitree-{Go2,As2}-Blind exports (branch
go2-blind). Same 47-D observation, no depth, no recurrent memory: the ONNX
takes one `obs` input that IS the 10-frame term-major history (oldest
first, reset backfilled -- TermHistory) and returns `actions`; its command
envelope is the blind joystick range (deploy.yaml scale_to_range). Its
exports carry no "contract" key and are recognised by that I/O signature.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import contract as C
from .contract import CONTROL_DT, PROPRIO_HISTORY_LENGTH, projected_gravity

STUDENT_CONTRACTS = (
    "go2_depth_v2",
    "as2_depth_v1",
    "b2_depth_v1",
    "go2w_depth_v1",
    "go2w_depth_v2",
    "b2w_depth_v2",
    "as2w_depth_v2",
)
HEADING_SUFFIX = "_heading"
HEADING_CONTRACTS = tuple(
    name + HEADING_SUFFIX
    for name in ("go2_depth_v2", "as2_depth_v1", "b2_depth_v1",
                 "go2w_depth_v2", "b2w_depth_v2", "as2w_depth_v2"))
# Students of go2-blind's legged finals (unitree_rl_mjlab depth-blind, 2026-10-27):
# the heading students' inputs with the speed clock and hip-0 stance of their
# blind teachers, and navcore's camera (its frame and mount in the metadata,
# DepthSpec).
NAVCORE_CONTRACTS = ("go2_depth_v3", "as2_depth_v3", "b2_depth_v3")
# A stereo camera reads nothing nearer than its minimum range: training's
# real-camera randomization (depth_env_cfg REAL_CAMERA_DEPTH min_range_m) draws
# it per episode from 0.05-0.15 m and reads those pixels as misses. The sim
# takes the far end: with exact depth down to 5 cm, AS2 and B2 students held
# against a wall fell (2026-10-28). navcore's own value is not measured yet.
NAVCORE_MIN_RANGE_M = 0.15
OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
)
HEADING_TERM = "heading_error"

GAIT_PERIOD_S = 0.6
COMMAND_DEADZONE = 0.1
# Training command envelope (unitree_rl_mjlab depth FINAL_VEL_RANGES, all
# robots).
COMMAND_LOW = np.asarray((-0.5, -0.5, -1.2), dtype=np.float32)
COMMAND_HIGH = np.asarray((1.0, 0.5, 1.2), dtype=np.float32)
# Blind joystick envelope (go2-blind deploy.yaml command ranges).
BLIND_COMMAND_LOW = np.asarray((-1.0, -0.6, -1.5), dtype=np.float32)
BLIND_COMMAND_HIGH = np.asarray((1.5, 0.6, 1.5), dtype=np.float32)


@dataclass(frozen=True)
class ActionGroup:
    """One action term: `scale * action` is a joint position offset from
    the default pose ("position") or a joint velocity target ("velocity")."""

    kind: str
    scale: float
    joints: tuple[str, ...]


def parse_action_layout(text: str) -> tuple[ActionGroup, ...]:
    """"joint_pos[position]*0.25:FL_hip_joint,...;wheel_vel[velocity]*10:..."."""
    groups = []
    for term in filter(None, text.split(";")):
        head, joints = term.split(":")
        kind = head[head.index("[") + 1:head.index("]")]
        if kind not in ("position", "velocity"):
            raise ValueError(f"unknown action kind {kind!r} in {term!r}")
        groups.append(ActionGroup(kind, float(head.split("*")[1]),
                                  tuple(joints.split(","))))
    return tuple(groups)


def phase(control_step: int, command: Sequence[float]) -> np.ndarray:
    if float(np.linalg.norm(np.asarray(command, dtype=np.float32))) < COMMAND_DEADZONE:
        return np.zeros(2, dtype=np.float32)
    t = (control_step * CONTROL_DT) % GAIT_PERIOD_S / GAIT_PERIOD_S
    return np.asarray((math.sin(2.0 * math.pi * t), math.cos(2.0 * math.pi * t)),
                      dtype=np.float32)


@dataclass(frozen=True)
class DepthSpec:
    """What a depth_history frame holds (ONNX metadata depth_camera, depth_size,
    depth_focal_px, depth_kind, depth_max_m, depth_camera_pos / _quat):
    width x height pixels of a pinhole with its principal point at the centre,
    optical Z (a stereo camera's output) or ray distance, normalized to
    [DEPTH_MIN_M, max_m] / max_m, from a camera at camera_pos / camera_quat in
    base_link (MuJoCo camera axes); nearer than min_range_m reads as a miss.
    Exports without the metadata use tinynav's PIE camera (PIE_DEPTH:
    sim/contract.py, its mount the robot's)."""

    name: str
    width: int
    height: int
    focal_px: float
    optical_z: bool
    max_m: float
    camera_pos: tuple[float, float, float] | None = None
    camera_quat: tuple[float, float, float, float] | None = None
    min_range_m: float = 0.0

    @property
    def pie(self) -> bool:
        return self.name == "pie"

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def history_shape(self) -> tuple[int, int, int]:
        return (C.DEPTH_HISTORY_LENGTH, self.height, self.width)

    @property
    def fovy_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(0.5 * self.height / self.focal_px))

    def preprocess(self, depth_z: np.ndarray) -> np.ndarray:
        """A native MuJoCo optical-Z render of this camera (0 = miss) -> one
        frame (1, height, width): PIE's crop / ray conversion / blur, or a
        stereo camera's Z as is (misses, anything nearer than min_range_m and
        anything past max_m read max_m)."""
        if self.pie:
            return C.preprocess_depth_z(depth_z)
        if not self.optical_z:
            raise ValueError(f"depth camera {self.name!r}: only optical-Z frames are rendered")
        depth = np.asarray(depth_z, dtype=np.float32)
        if depth.shape != self.frame_shape:
            raise ValueError(f"raw depth shape is {depth.shape}; expected {self.frame_shape}")
        seen = np.isfinite(depth) & (depth > 0.0) & (depth >= self.min_range_m)
        depth = np.where(seen, depth, self.max_m)
        frame = np.clip(depth, C.DEPTH_MIN_M, self.max_m) / self.max_m
        return np.ascontiguousarray(frame[None], dtype=np.float32)


PIE_DEPTH = DepthSpec(
    "pie", C.DEPTH_WIDTH, C.DEPTH_HEIGHT,
    0.5 * C.DEPTH_HEIGHT / math.tan(math.radians(C.DEPTH_FOVY_DEG) / 2), False, C.DEPTH_MAX_M)


def depth_spec(metadata: dict[str, str]) -> DepthSpec:
    """The depth frame an export was trained on (DepthSpec), PIE without metadata."""
    if "depth_camera" not in metadata:
        return PIE_DEPTH
    width, height = (int(v) for v in metadata["depth_size"].split(","))
    pos = metadata.get("depth_camera_pos")
    quat = metadata.get("depth_camera_quat")
    return DepthSpec(
        metadata["depth_camera"], width, height, float(metadata["depth_focal_px"]),
        metadata.get("depth_kind", "ray") == "optical_z",
        float(metadata.get("depth_max_m", C.DEPTH_MAX_M)),
        tuple(float(v) for v in pos.split(",")) if pos else None,
        tuple(float(v) for v in quat.split(",")) if quat else None,
        NAVCORE_MIN_RANGE_M if metadata["depth_camera"] == "navcore" else 0.0)


class GaitClock:
    """go2-blind's speed clock for one robot (mdp.gait_clock / phase, deploy
    gait_phase; ONNX metadata gait_period, gait_ref_speed, gait_foot_xy):
    above ref_speed the period shortens to period * ref_speed / speed, the
    speed being the fastest foot's under the command (foot_xy, the feet's
    nominal base-frame xy: |(vx - wz y, vy + wz x)|; planar command speed
    without it). The phase is 0 at the first control step after a reset and
    advances CONTROL_DT over the period of each later step's command; it
    reads zero below the command deadzone unless the robot marches in place.
    Call once per control step."""

    def __init__(self, period: float, ref_speed: float,
                 foot_xy: np.ndarray | None = None) -> None:
        self.period = float(period)
        self.ref_speed = float(ref_speed)
        self.foot_xy = None if foot_xy is None else np.asarray(foot_xy, np.float64).reshape(-1, 2)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self._started = False

    def current_period(self, command: Sequence[float]) -> float:
        vx, vy, wz = (float(v) for v in np.asarray(command).reshape(3))
        if self.foot_xy is None:
            speed = math.hypot(vx, vy)
        else:
            speed = max(math.hypot(vx - wz * y, vy + wz * x) for x, y in self.foot_xy)
        return self.period * min(1.0, self.ref_speed / max(speed, 1e-6))

    def __call__(self, command: Sequence[float], march: bool = False) -> np.ndarray:
        if self._started:
            self.value = (self.value + CONTROL_DT / self.current_period(command)) % 1.0
        self._started = True
        if float(np.linalg.norm(np.asarray(command, dtype=np.float32))) < COMMAND_DEADZONE and not march:
            return np.zeros(2, dtype=np.float32)
        return np.asarray((math.sin(2.0 * math.pi * self.value),
                           math.cos(2.0 * math.pi * self.value)), dtype=np.float32)


HEADING_MAX_ERROR = 0.5
HEADING_SETTLE_STEPS = 2


def yaw_from_quaternion(quaternion_wxyz: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class HeadingError:
    """heading_error state for one robot: call once per control step."""

    def __init__(self, max_error: float = HEADING_MAX_ERROR) -> None:
        self.max_error = float(max_error)
        self.reset()

    def reset(self) -> None:
        self.target = 0.0
        self.settle = HEADING_SETTLE_STEPS

    def __call__(self, quaternion_wxyz: Sequence[float], yaw_rate_command: float) -> float:
        yaw = yaw_from_quaternion(quaternion_wxyz)
        if self.settle > 0:
            self.target = yaw
            self.settle -= 1
        else:
            self.target += float(yaw_rate_command) * CONTROL_DT
        error = math.atan2(math.sin(self.target - yaw), math.cos(self.target - yaw))
        error = min(max(error, -self.max_error), self.max_error)
        self.target = yaw + error
        return error


class TermHistory:
    """Term-major, oldest-to-newest observation history (MJLab keeps one
    circular buffer per term and concatenates the flattened buffers -- not
    frames.reshape(-1)). sim/contract.py ProprioHistory, any term dims."""

    def __init__(self, term_dims: Sequence[int],
                 length: int = PROPRIO_HISTORY_LENGTH) -> None:
        self.term_dims = tuple(int(d) for d in term_dims)
        self.obs_dim = sum(self.term_dims)
        self.length = int(length)
        self._frames: deque[np.ndarray] = deque(maxlen=self.length)

    def reset(self) -> None:
        self._frames.clear()

    def append(self, observation: Sequence[float]) -> None:
        frame = np.asarray(observation, dtype=np.float32).reshape(self.obs_dim)
        if not np.all(np.isfinite(frame)):
            raise FloatingPointError("Cannot append non-finite proprioception.")
        if not self._frames:
            for _ in range(self.length):
                self._frames.append(frame.copy())
            return
        self._frames.append(frame.copy())

    @property
    def array(self) -> np.ndarray:
        if len(self._frames) != self.length:
            raise RuntimeError("Append one observation before reading history.")
        frames = np.stack(tuple(self._frames), axis=0)
        chunks = []
        start = 0
        for dim in self.term_dims:
            chunks.append(frames[:, start:start + dim].reshape(-1))
            start += dim
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def _floats(metadata: dict[str, str], key: str) -> np.ndarray:
    try:
        return np.asarray([float(v) for v in metadata[key].split(",") if v],
                          dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"ONNX metadata is missing {key!r}.") from exc


class StudentContract:
    """Depth-student observation/action contract, built from ONNX metadata."""

    def __init__(self, metadata: dict[str, str], name: str | None = None,
                 command_low=COMMAND_LOW, command_high=COMMAND_HIGH) -> None:
        self.name = name or metadata["contract"]
        if name is None and self.name not in STUDENT_CONTRACTS + HEADING_CONTRACTS + NAVCORE_CONTRACTS:
            raise ValueError(f"Unknown ONNX policy contract {self.name!r}.")
        if self.name in HEADING_CONTRACTS + NAVCORE_CONTRACTS:
            # go2-blind's command ranges (the heading recipe trains on them)
            command_low, command_high = BLIND_COMMAND_LOW, BLIND_COMMAND_HIGH
        self.joint_names = tuple(metadata.get("joint_names", "").split(","))
        self.default_joint_pos = _floats(metadata, "default_joint_pos")
        if self.default_joint_pos.shape != (len(self.joint_names),):
            raise ValueError("ONNX default_joint_pos does not match joint_names.")
        if "action_layout" in metadata:
            self.action_groups = parse_action_layout(metadata["action_layout"])
        else:
            scale = _floats(metadata, "action_scale")
            if scale.shape != (1,):
                raise ValueError("Per-joint action_scale needs an action_layout.")
            self.action_groups = (ActionGroup("position", float(scale[0]),
                                              self.joint_names),)
        for group in self.action_groups:
            unknown = set(group.joints) - set(self.joint_names)
            if unknown:
                raise ValueError(f"action_layout names unknown joints {sorted(unknown)}")
        self.position_joints = tuple(
            j for g in self.action_groups if g.kind == "position" for j in g.joints)
        self._pos_index = np.asarray(
            [self.joint_names.index(j) for j in self.position_joints])
        self.NUM_ACTIONS = sum(len(g.joints) for g in self.action_groups)
        names = tuple(filter(None, metadata.get(
            "observation_names", ",".join(OBSERVATION_NAMES)).split(",")))
        known = OBSERVATION_NAMES + (HEADING_TERM,)
        if [n for n in known if n in names] != list(names):
            raise ValueError(f"Unsupported observation terms {names}.")
        self.heading = HEADING_TERM in names
        if self.heading != (self.name in HEADING_CONTRACTS + NAVCORE_CONTRACTS) and name is None:
            raise ValueError(f"Contract {self.name!r} with observation terms {names}.")
        self.OBSERVATION_NAMES = names
        dims = {"base_ang_vel": 3, "projected_gravity": 3, "command": 3,
                "phase": 2, "joint_pos": len(self.position_joints),
                "joint_vel": len(self.joint_names), "actions": self.NUM_ACTIONS,
                HEADING_TERM: 1}
        self.OBSERVATION_TERM_DIMS = tuple(dims[n] for n in names)
        brake = metadata.get("wheel_brake_rad_s")
        self.wheel_brake_rad_s = float(brake) if brake else None
        # the depth_history frame it was trained on (PIE without metadata)
        self.depth = depth_spec(metadata)
        # go2-blind's speed clock (GaitClock); without gait_ref_speed the fixed
        # 0.6 s clock of phase().
        ref_speed = metadata.get("gait_ref_speed")
        self.gait_ref_speed = float(ref_speed) if ref_speed else None
        self.gait_period = float(metadata.get("gait_period") or GAIT_PERIOD_S)
        feet = metadata.get("gait_foot_xy")
        self.gait_foot_xy = (np.asarray([[float(v) for v in f.split(",")] for f in feet.split(";")])
                             if feet else None)
        # the policy was trained to march in place on a zero command when told to
        self.gait_march = metadata.get("gait_march", "0") in ("1", "true", "True")
        self.OBS_DIM = sum(self.OBSERVATION_TERM_DIMS)
        self.PROPRIO_HISTORY_DIM = self.OBS_DIM * PROPRIO_HISTORY_LENGTH
        self.COMMAND_LOW = np.asarray(command_low, dtype=np.float32)
        self.COMMAND_HIGH = np.asarray(command_high, dtype=np.float32)

    def ProprioHistory(self) -> TermHistory:  # noqa: N802 -- contract surface
        return TermHistory(self.OBSERVATION_TERM_DIMS)

    def HeadingError(self) -> HeadingError | None:  # noqa: N802 -- contract surface
        """heading_error state for one robot, None without the term."""
        return HeadingError() if self.heading else None

    def GaitClock(self) -> GaitClock | None:  # noqa: N802 -- contract surface
        """Speed-clock state for one robot, None for the fixed clock
        (control_step) or without a phase term."""
        if self.gait_ref_speed is None or "phase" not in self.OBSERVATION_NAMES:
            return None
        return GaitClock(self.gait_period, self.gait_ref_speed, self.gait_foot_xy)

    def shape_command(self, command: Sequence[float]) -> np.ndarray:
        """Clip to the training envelope and apply the training deadzone."""
        cmd = np.clip(np.asarray(command, dtype=np.float32).reshape(3),
                      self.COMMAND_LOW, self.COMMAND_HIGH)
        # float32 on both sides, as torch compares the command tensor
        if np.linalg.norm(cmd) <= np.float32(COMMAND_DEADZONE):
            cmd[:] = 0.0
        return cmd

    def build_proprioception(
        self,
        base_ang_vel: Sequence[float],
        quaternion_wxyz: Sequence[float],
        command: Sequence[float],
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        last_action: Sequence[float],
        control_step: int,
        heading_error: float | None = None,
        gait_phase: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Observation; joint_pos/joint_vel in joint_names order, `command`
        already shaped, `heading_error` this step's HeadingError value
        (heading contracts only) and `gait_phase` this step's GaitClock value
        (speed-clock exports only)."""
        if self.heading and heading_error is None:
            raise ValueError(f"{self.name} needs heading_error.")
        if (self.gait_ref_speed is not None and "phase" in self.OBSERVATION_NAMES
                and gait_phase is None):
            raise ValueError(f"{self.name} needs gait_phase (GaitClock).")
        cmd = np.asarray(command, dtype=np.float32).reshape(3)
        rel = (np.asarray(joint_pos, dtype=np.float32).reshape(-1)
               - self.default_joint_pos)
        values = {
            "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float32).reshape(3),
            "projected_gravity": projected_gravity(quaternion_wxyz),
            "command": cmd,
            "phase": (phase(control_step, cmd) if gait_phase is None
                      else np.asarray(gait_phase, dtype=np.float32).reshape(2)),
            "joint_pos": rel[self._pos_index],
            "joint_vel": np.asarray(joint_vel, dtype=np.float32).reshape(
                len(self.joint_names)),
            "actions": np.asarray(last_action, dtype=np.float32).reshape(
                self.NUM_ACTIONS),
        }
        if self.heading:
            values[HEADING_TERM] = np.asarray((heading_error,), dtype=np.float32)
        observation = np.concatenate(
            [values[n] for n in self.OBSERVATION_NAMES]).astype(np.float32, copy=False)
        if observation.shape != (self.OBS_DIM,) or not np.all(np.isfinite(observation)):
            raise FloatingPointError(
                f"Invalid {self.name} proprioception with shape {observation.shape}.")
        return np.ascontiguousarray(observation)


class BlindContract(StudentContract):
    """Blind policy (go2-blind Unitree-<Robot>-Blind exports): the student
    observation with the blind joystick envelope. The wheel-legged ones have
    no phase and come through mujoco/scripts/import_blind_policy.py, which
    writes their policy joint order, action layout and brake into the
    metadata."""

    def __init__(self, metadata: dict[str, str]) -> None:
        super().__init__(metadata, name="blind", command_low=BLIND_COMMAND_LOW,
                         command_high=BLIND_COMMAND_HIGH)


def contract_for(metadata: dict[str, str],
                 input_names: Sequence[str] = ()) -> StudentContract:
    """ONNX metadata (+ input names) -> contract object. Depth students name
    their contract; blind exports are the single-`obs` signature with the
    student observation terms; anything else (e.g. the retired PIE
    policies) is refused."""
    if "contract" in metadata:
        return StudentContract(metadata)
    no_phase = tuple(n for n in OBSERVATION_NAMES if n != "phase")
    if (tuple(input_names) == ("obs",)
            and metadata.get("observation_names") in tuple(
                ",".join(names + extra)
                for names in (OBSERVATION_NAMES, no_phase)
                for extra in ((), (HEADING_TERM,)))):
        return BlindContract(metadata)
    raise ValueError("ONNX is neither a depth-student export (metadata "
                     f"'contract' in {', '.join(STUDENT_CONTRACTS + HEADING_CONTRACTS + NAVCORE_CONTRACTS)}) "
                     "nor a blind export (single 'obs' input).")


__all__ = [
    "ActionGroup",
    "BlindContract",
    "COMMAND_HIGH",
    "COMMAND_LOW",
    "DepthSpec",
    "GaitClock",
    "HEADING_CONTRACTS",
    "HeadingError",
    "NAVCORE_CONTRACTS",
    "PIE_DEPTH",
    "STUDENT_CONTRACTS",
    "StudentContract",
    "TermHistory",
    "contract_for",
    "depth_spec",
    "parse_action_layout",
    "phase",
]
