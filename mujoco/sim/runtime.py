#!/usr/bin/env python3
"""PIE 50 Hz control runtime over a native-MuJoCo plant.

Adapted from parkour_mjlab deploy/pie/sim2sim/go2_pie_sim2sim.py
(train/orient-ablation): depth push every DEPTH_UPDATE_PERIOD_STEPS ticks
(10 Hz), observation -> term-major history -> ONNX -> ctrl = q_target ->
PHYSICS_STEPS_PER_CONTROL sub-steps (PD lives in the actuator model).
The command source is injected (keyboard for the live host, scripted for
headless smokes); the plant is pie_model's map3 or flat builder.

The policy's contract (sim/contract_v2.py) decides the observation layout and
which actuators its actions drive (leg position targets, wheel velocity
targets), so any registered robot runs here; a policy whose joints are not
the plant's is refused. Blind policies (no depth input) skip the camera.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from . import contract as C
from .plant import Bindings, reset_to_spawn

# Joint-target safety clamp around the default pose, as the real-robot deploy
# applies it (go2-blind deploy.yaml: default +- 2.0 rad = raw action +- 8).
# Training had none and it bites on ~0.005% of steps; last_action stays the
# raw policy output.
ACTION_CLIP_RAD = 2.0


class DepthCamera:
    """Native MuJoCo optical-Z renderer matching the raw PIE camera."""

    def __init__(self, model: mujoco.MjModel, camera_id: int) -> None:
        self._model = model
        self._camera_id = camera_id
        self._renderer = mujoco.Renderer(
            model, height=C.DEPTH_HEIGHT, width=C.DEPTH_RAW_WIDTH
        )
        self._renderer.enable_depth_rendering()
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[:3] = 1   # visuals + map3 boxes (group 0/2)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self._far_clip = float(model.vis.map.zfar * model.stat.extent)
        near_clip = float(model.vis.map.znear * model.stat.extent)
        if not 0.0 < near_clip < C.DEPTH_MIN_M:
            raise ValueError(
                f"depth near plane {near_clip} must be below {C.DEPTH_MIN_M}"
            )
        if self._far_clip * 0.99 <= C.DEPTH_MAX_M:
            raise ValueError("depth far plane does not exceed the PIE cutoff")

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.update_scene(
            data, camera=self._camera_id, scene_option=self._scene_option
        )
        depth = np.asarray(self._renderer.render(), dtype=np.float32)
        if depth.shape != (C.DEPTH_HEIGHT, C.DEPTH_RAW_WIDTH):
            raise ValueError(f"rendered depth has invalid shape {depth.shape}")
        misses = (
            ~np.isfinite(depth)
            | (depth <= 0.0)
            | (depth >= self._far_clip * 0.99)
        )
        result = depth.copy()
        result[misses] = 0.0
        return np.ascontiguousarray(result, dtype=np.float32)

    def close(self) -> None:
        self._renderer.close()


def _sensor_address(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"model is missing sensor {name!r}")
    return int(model.sensor_adr[sid])


class PieRuntime:
    def __init__(self, model: mujoco.MjModel, b: Bindings,
                 data: mujoco.MjData, policy, depth_camera: DepthCamera,
                 cmd_source, spawn_pos=None, spawn_yaw_deg=None,
                 substep_hook=None) -> None:
        self.model = model
        self.b = b
        self.data = data
        self.policy = policy
        self.camera = depth_camera
        self.cmd_source = cmd_source
        self._spawn_pos = spawn_pos
        self._spawn_yaw_deg = spawn_yaw_deg
        # Optional per-physics-substep callback (called right after each
        # mj_step, inside the control thread -- safe to read MjData here,
        # NOT from other threads). Used by the transport face to sample the
        # IMU and base pose at the full 200 Hz physics rate.
        self.substep_hook = substep_hook
        # Contract of the loaded policy (PIE or a depth student).
        self.contract = policy.contract
        self.uses_depth = bool(getattr(policy, "uses_depth", True))
        self._bind_contract()
        self.last_action = np.zeros(self.contract.NUM_ACTIONS, dtype=np.float32)
        self.proprio_history = self.contract.ProprioHistory()
        self.depth_history = C.DepthHistory()
        self.control_step = 0
        self.latest_raw_depth = None
        self.command = np.zeros(3, dtype=np.float32)   # last command the policy saw
        self.policy_ms = 0.0
        self.physics_ms = 0.0
        self._last_data_time = float(data.time)

    def _bind_contract(self) -> None:
        """Map the contract's joints and action groups onto the plant, and
        refuse a policy trained for another robot (joints, default pose or
        PD gains differ)."""
        k, b = self.contract, self.b
        names = b.joint_names
        if sorted(k.joint_names) != sorted(names):
            raise ValueError(
                f"policy contract {k.name!r} joints {k.joint_names} do not "
                f"match robot {b.robot.name!r} joints {names}")
        idx = np.asarray([names.index(j) for j in k.joint_names])
        self._qpos_adr = b.joint_qpos_adr[idx]
        self._dof_adr = b.joint_dof_adr[idx]
        default = np.asarray(k.default_joint_pos, dtype=np.float32)
        if not np.allclose(default, b.default_joint_pos[idx], atol=1e-3):
            raise ValueError(f"policy {k.name!r} default pose does not match "
                             f"robot {b.robot.name!r}")
        meta = getattr(self.policy, "metadata", {})
        gains = {key: [float(v) for v in meta[key].split(",")]
                 for key in ("joint_stiffness", "joint_damping") if key in meta}
        # (actuator ids, slice of the action vector, scale, default or None)
        self._action_map = []
        start = 0
        for group in k.action_groups:
            jidx = [k.joint_names.index(j) for j in group.joints]
            ids = b.actuator_ids[[names.index(j) for j in group.joints]]
            stop = start + len(ids)
            offset = default[jidx] if group.kind == "position" else None
            self._action_map.append((ids, slice(start, stop),
                                     np.float32(group.scale), offset))
            start = stop
            if group.kind == "position" and len(gains) == 2:
                kp = np.asarray(gains["joint_stiffness"])[jidx]
                kd = np.asarray(gains["joint_damping"])[jidx]
                if not (np.allclose(self.model.actuator_gainprm[ids, 0], kp, atol=1e-3)
                        and np.allclose(-self.model.actuator_biasprm[ids, 2], kd,
                                        atol=1e-3)):
                    raise ValueError(f"policy {k.name!r} PD gains do not match "
                                     f"robot {b.robot.name!r} actuators")

    # -- state -------------------------------------------------------------
    def base_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[self.b.root_qpos_adr:self.b.root_qpos_adr + 3])

    def base_uprightness(self) -> float:
        root = self.b.root_qpos_adr
        quat = self.data.qpos[root + 3:root + 7]
        gz = C.projected_gravity(quat)[2]
        return float(-gz)   # 1.0 upright, ~0 on its side, -1 upside down

    def is_fallen(self) -> bool:
        return self.base_uprightness() < 0.5

    def reset(self) -> None:
        reset_to_spawn(self.model, self.data, self.b, settle=True)
        self.last_action.fill(0.0)
        self.proprio_history.reset()
        self.depth_history.reset()
        self.policy.reset()
        self.control_step = 0
        self.latest_raw_depth = None

    # -- observation (mirrors upstream build_observation) -------------------
    def build_observation(self, command: np.ndarray) -> np.ndarray:
        root = self.b.root_qpos_adr
        quaternion = self.data.qpos[root + 3:root + 7]
        angular_velocity = self.data.sensordata[
            self.b.gyro_adr:self.b.gyro_adr + 3
        ]
        joint_pos = self.data.qpos[self._qpos_adr]
        joint_vel = self.data.qvel[self._dof_adr]
        # Students: training command envelope + deadzone, and the gait phase
        # clock counted from the last reset (control_step). PIE: unshaped.
        self.command = self.contract.shape_command(command)
        return self.contract.build_proprioception(
            angular_velocity, quaternion, self.command,
            joint_pos, joint_vel, self.last_action, self.control_step,
        )

    def depth_frame(self) -> np.ndarray | None:
        """Newest policy depth frame (60x86, ray depth / DEPTH_MAX_M), or
        None for a blind policy / before the first capture."""
        if not self.uses_depth or self.latest_raw_depth is None:
            return None
        return self.depth_history.array[-1]

    def _resettle_after_viewer_reset(self) -> None:
        """Recover from the viewer's Reset button: mj_resetData jumps to
        model.qpos0 (the COMPILE-time attach frame -- a different spot and
        straight legs), so re-place the dog at the exact startup spawn and
        settle, then clear policy state (stale GRU/histories)."""
        reset_to_spawn(self.model, self.data, self.b, settle=True,
                       pos=self._spawn_pos, yaw_deg=self._spawn_yaw_deg)
        self.last_action.fill(0.0)
        self.proprio_history.reset()
        self.depth_history.reset()
        self.policy.reset()
        self.control_step = 0
        self.latest_raw_depth = None

    # -- one 50 Hz control tick ---------------------------------------------
    def tick(self) -> None:
        # The viewer's Reset button mj_resetData's the shared state behind
        # our back: data.time jumps backward. The GRU/history are then stale
        # and the joints sit at qpos0 (straight legs) -- resettle and clear.
        if float(self.data.time) < self._last_data_time - 1e-9:
            print(f"[RT] viewer reset detected (t {self._last_data_time:.1f}"
                  f"->{float(self.data.time):.1f}), resettling", flush=True)
            self._resettle_after_viewer_reset()
        self._last_data_time = float(self.data.time)

        if self.cmd_source.consume_reset():
            self.reset()
            return

        if (self.uses_depth
                and self.control_step % C.DEPTH_UPDATE_PERIOD_STEPS == 0):
            raw_depth = self.camera.capture(self.data)
            self.latest_raw_depth = raw_depth
            self.depth_history.append(C.preprocess_depth_z(raw_depth))

        observation = self.build_observation(self.cmd_source.vector())
        self.proprio_history.append(observation)
        t0 = time.perf_counter()
        action = self.policy(
            observation,
            self.proprio_history.array,
            self.depth_history.array if self.uses_depth else None,
        )
        self.policy_ms = (time.perf_counter() - t0) * 1e3
        for ids, part, scale, offset in self._action_map:
            # position: default pose + scale * action; velocity: scale * action
            target = scale * action[part]
            if offset is not None:
                target = np.clip(offset + target, offset - ACTION_CLIP_RAD,
                                 offset + ACTION_CLIP_RAD)
            if not np.all(np.isfinite(target)):
                raise FloatingPointError("policy actuator target is non-finite")
            self.data.ctrl[ids] = target
        t0 = time.perf_counter()
        for _ in range(C.PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)
            if self.substep_hook is not None:
                self.substep_hook(self.data)
        self.physics_ms = (time.perf_counter() - t0) * 1e3
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
                np.isfinite(self.data.qvel)):
            raise FloatingPointError("MuJoCo state became non-finite")
        self.last_action = action
        self.control_step += 1
