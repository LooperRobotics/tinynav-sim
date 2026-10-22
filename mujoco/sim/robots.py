#!/usr/bin/env python3
"""Robot registry: the bodies the sim can drive and the policies per body.

One entry per robot model. The body is a training MJCF
(assets/<name>/<name>.xml, exported by the RL training repo with its
actuators and collision setup -- see that file's header), its visual meshes
are the decimated MSH set in the model/ asset bundle (model/<name>/assets).
The old PIE plant (assets/go2/go2_pie.xml with the sim/contract.py actuators,
`pie_plant`) is no longer registered.

Policies are ONNX files in assets/policy/: "perception" reads the front
depth camera, "blind" runs on proprioception alone. A robot is selectable
when its body (MJCF + meshes) and at least one policy are present;
`missing()` says what is not.

The depth camera mount is the training one (base_link frame). Only Go2's is
a measured extrinsic; the others are placed the same way in front of each
head and are estimates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent          # .../mujoco/sim
ASSETS = _HERE.parent / "assets"
POLICY_DIR = ASSETS / "policy"
REPO = _HERE.parents[1]                          # .../tinynav-sim

# PIE/Go2 depth camera mount (sim/contract.py DEPTH_CAMERA_*), shared by the
# Go2-bodied robots.
_GO2_CAMERA = ((0.345, 0.0, 0.07), (0.5792280, 0.4055798, -0.4055798, -0.5792280))
_CAMERA_QUAT = _GO2_CAMERA[1]

# Training default poses (unitree_rl_mjlab robot INIT_STATE joint_pos, go2-blind's
# robot models): Go2 / AS2 / AS2-W, Go2-W / B2-W, B2.
_GO2_POSE = {r".*thigh_joint": 0.9, r".*calf_joint": -1.8,
             r".*R_hip_joint": 0.1, r".*L_hip_joint": -0.1, r".*_wheel_joint": 0.0}
_WHEEL_POSE = {r".*thigh_joint": 0.8, r".*calf_joint": -1.5, r".*hip_joint": 0.0,
               r".*_wheel_joint": 0.0}
_B2_POSE = {r"F._thigh_joint": 0.8, r"R._thigh_joint": 1.0, r".*calf_joint": -1.5,
            r".*L_hip_joint": 0.1, r".*R_hip_joint": -0.1}


@dataclass(frozen=True)
class Robot:
    name: str
    label: str
    camera_pos: tuple[float, float, float]
    camera_quat: tuple[float, float, float, float]
    spawn_drop: float
    """Base height above the ground at spawn (m); the training init height."""
    default_pose: dict[str, float] = field(default_factory=dict)
    """Joint-name regex -> default angle (first match wins)."""
    perception: str | None = None
    blind: str | None = None
    wheeled: bool = False
    pie_plant: bool = False
    """go2 only: the PIE plant (go2_pie.xml + sim/contract.py actuators)."""

    @property
    def mjcf(self) -> Path:
        return ASSETS / self.name / f"{self.name}.xml"

    def mesh_dir(self) -> Path | None:
        """Visual meshes, model/ bundle search order: $ROBOT_ASSETS, the
        in-repo model/ bundle, the legacy robot-assets checkout."""
        roots = [Path(os.environ["ROBOT_ASSETS"])] if os.environ.get("ROBOT_ASSETS") else []
        roots += [REPO / "model", Path.home() / "workspace" / "dm" / "robot-assets"]
        for root in roots:
            d = root / self.name / "assets"
            if d.is_dir():
                return d
        return None

    def policy_path(self, kind: str) -> Path | None:
        name = {"perception": self.perception, "blind": self.blind}[kind]
        return POLICY_DIR / name if name else None

    def has_policy(self, kind: str) -> bool:
        path = self.policy_path(kind)
        return path is not None and path.is_file()

    def default_joint_pos(self, joint_names) -> np.ndarray:
        out = []
        for joint in joint_names:
            for pattern, value in self.default_pose.items():
                if re.fullmatch(pattern, joint):
                    out.append(value)
                    break
            else:
                raise ValueError(f"{self.name}: no default angle for joint {joint!r}")
        return np.asarray(out, dtype=np.float64)

    def missing(self) -> list[str]:
        """What keeps this robot from being driven (empty = selectable)."""
        out = []
        if not self.pie_plant and not self.mjcf.is_file():
            out.append("no MJCF")
        if self.mesh_dir() is None:
            out.append("no meshes (model.zip)")
        if not (self.has_policy("perception") or self.has_policy("blind")):
            out.append("no policy yet")
        return out


ROBOTS: dict[str, Robot] = {
    # Plants: go2-blind's robot models (unitree_rl_mjlab export_sim_robot.py),
    # each driven by its heading-recipe depth student (the old models' students
    # are not driven on them).
    "go2": Robot("go2", "Go2", *_GO2_CAMERA, spawn_drop=0.32,
                 default_pose=_GO2_POSE, perception="go2_depth_v2_heading.onnx",
                 blind="go2_blind_v2.onnx"),
    "go2w": Robot("go2w", "Go2-W", *_GO2_CAMERA, spawn_drop=0.43,
                  default_pose=_WHEEL_POSE, perception="go2w_depth_v2_heading.onnx",
                  blind="go2w_blind_v0.onnx", wheeled=True),
    "as2": Robot("as2", "AS2", (0.325, 0.0, 0.075), _CAMERA_QUAT,
                 spawn_drop=0.307, default_pose=_GO2_POSE,
                 perception="as2_depth_v1_heading_cap.onnx", blind="as2_blind_v1.onnx"),
    "as2w": Robot("as2w", "AS2-W", (0.325, 0.0, 0.075), _CAMERA_QUAT,
                  spawn_drop=0.37, default_pose=_GO2_POSE,
                  perception="as2w_depth_v2_heading.onnx", blind="as2w_blind_v0.onnx",
                  wheeled=True),
    "b2": Robot("b2", "B2", (0.445, 0.0, 0.05), _CAMERA_QUAT,
                spawn_drop=0.53, default_pose=_B2_POSE,
                perception="b2_depth_v1_heading_cap.onnx", blind="b2_blind_v0.onnx"),
    "b2w": Robot("b2w", "B2-W", (0.445, 0.0, 0.05), _CAMERA_QUAT,
                 spawn_drop=0.645, default_pose=_WHEEL_POSE,
                 perception="b2w_depth_v2_heading.onnx", blind="b2w_blind_v0.onnx",
                 wheeled=True),
}
DEFAULT_ROBOT = "go2"


def get_robot(name: str) -> Robot:
    try:
        return ROBOTS[name]
    except KeyError:
        raise SystemExit(f"unknown robot {name!r}; choose {sorted(ROBOTS)}") from None


def available() -> list[str]:
    return [name for name, robot in ROBOTS.items() if not robot.missing()]


__all__ = ["DEFAULT_ROBOT", "ROBOTS", "Robot", "available", "get_robot"]
