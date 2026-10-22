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
        if name is None and self.name not in STUDENT_CONTRACTS + HEADING_CONTRACTS:
            raise ValueError(f"Unknown ONNX policy contract {self.name!r}.")
        if self.name in HEADING_CONTRACTS:
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
        if self.heading != (self.name in HEADING_CONTRACTS) and name is None:
            raise ValueError(f"Contract {self.name!r} with observation terms {names}.")
        self.OBSERVATION_NAMES = names
        dims = {"base_ang_vel": 3, "projected_gravity": 3, "command": 3,
                "phase": 2, "joint_pos": len(self.position_joints),
                "joint_vel": len(self.joint_names), "actions": self.NUM_ACTIONS,
                HEADING_TERM: 1}
        self.OBSERVATION_TERM_DIMS = tuple(dims[n] for n in names)
        brake = metadata.get("wheel_brake_rad_s")
        self.wheel_brake_rad_s = float(brake) if brake else None
        self.OBS_DIM = sum(self.OBSERVATION_TERM_DIMS)
        self.PROPRIO_HISTORY_DIM = self.OBS_DIM * PROPRIO_HISTORY_LENGTH
        self.COMMAND_LOW = np.asarray(command_low, dtype=np.float32)
        self.COMMAND_HIGH = np.asarray(command_high, dtype=np.float32)

    def ProprioHistory(self) -> TermHistory:  # noqa: N802 -- contract surface
        return TermHistory(self.OBSERVATION_TERM_DIMS)

    def HeadingError(self) -> HeadingError | None:  # noqa: N802 -- contract surface
        """heading_error state for one robot, None without the term."""
        return HeadingError() if self.heading else None

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
    ) -> np.ndarray:
        """Observation; joint_pos/joint_vel in joint_names order, `command`
        already shaped and `heading_error` this step's HeadingError value
        (heading contracts only)."""
        if self.heading and heading_error is None:
            raise ValueError(f"{self.name} needs heading_error.")
        cmd = np.asarray(command, dtype=np.float32).reshape(3)
        rel = (np.asarray(joint_pos, dtype=np.float32).reshape(-1)
               - self.default_joint_pos)
        values = {
            "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float32).reshape(3),
            "projected_gravity": projected_gravity(quaternion_wxyz),
            "command": cmd,
            "phase": phase(control_step, cmd),
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
                     f"'contract' in {', '.join(STUDENT_CONTRACTS + HEADING_CONTRACTS)}) "
                     "nor a blind export (single 'obs' input).")


__all__ = [
    "ActionGroup",
    "BlindContract",
    "COMMAND_HIGH",
    "COMMAND_LOW",
    "HEADING_CONTRACTS",
    "HeadingError",
    "STUDENT_CONTRACTS",
    "StudentContract",
    "TermHistory",
    "contract_for",
    "parse_action_layout",
    "phase",
]
