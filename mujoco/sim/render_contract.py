#!/usr/bin/env python3
"""Native-MuJoCo contract camera pair: left/right grayscale + left depth.

Used by the pilot-emu contract face (mujoco/contract_face.py) for scenes with
no 3DGS splat (the parametric floorplan).  The model's `infra1_contract` /
`infra2_contract` cameras carry the looper contract's vertical fovy, so a
544x640 render has the same fx/cx/cy the face publishes in camera_info.

The renderer owns a PRIVATE MjData and an offscreen MuJoCo Renderer.  Create it
on the render thread (EGL contexts are thread-affine); the model may be built
on the main thread and only read here.
"""
from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np

from hil import CONTRACT_FX, CONTRACT_H, CONTRACT_W, optical


class MujocoContractRenderer:
    """Drop-in replacement for the gausscam render loop's frame producer.

    render(base_pos, base_quat_wxyz) -> (gray_l, gray_r, depth_mm_u16,
                                          R_opt, t_opt, jpeg_bytes|None)
    """

    def __init__(self, model: mujoco.MjModel, left_cam_id: int,
                 right_cam_id: int, width: int = CONTRACT_W,
                 height: int = CONTRACT_H, world_geomgroup: int = 0) -> None:
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._left = int(left_cam_id)
        self._right = int(right_cam_id)
        self.data = mujoco.MjData(model)
        self._rend = mujoco.Renderer(model, height=self.height,
                                     width=self.width)
        self._rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        # static world only: the dog rides the camera body and would read as
        # a false near obstacle if it entered the frame.  The floorplan's
        # floor/walls/well are group 0; the robot geoms are 2/3.
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 0
        self._opt.geomgroup[int(world_geomgroup)] = 1
        self._far = float(model.vis.map.zfar * model.stat.extent)

        fovy = float(model.cam_fovy[self._left])
        fxy = 0.5 * self.height / math.tan(math.radians(fovy) / 2.0)
        self.K = (fxy, self.width / 2.0, self.height / 2.0)
        if abs(fxy - CONTRACT_FX) > 0.01:
            raise ValueError(
                f"contract camera fx {fxy:.6f} != {CONTRACT_FX:.6f} -- "
                f"camera fovy {fovy:.6f} drifted")
        if abs(float(model.cam_fovy[self._right]) - fovy) > 1e-9:
            raise ValueError("contract left/right fovy differ")

    def _render_rgb(self, cam_id: int) -> np.ndarray:
        self._rend.update_scene(self.data, camera=cam_id,
                                scene_option=self._opt)
        return np.asarray(self._rend.render(), dtype=np.uint8)

    def render(self, base_pos, base_quat_wxyz, encode_jpeg: bool = True):
        """One frame at a MuJoCo WXYZ base pose (the emu ring's order)."""
        q = self.data.qpos
        q[0:3] = np.asarray(base_pos, dtype=float)
        q[3:7] = np.asarray(base_quat_wxyz, dtype=float)
        # Joints stay at qpos0 and the robot is masked; only the base pose
        # moves the contract cameras.
        mujoco.mj_forward(self.model, self.data)

        rgb_l = self._render_rgb(self._left)
        rgb_r = self._render_rgb(self._right)

        self._rend.enable_depth_rendering()
        self._rend.update_scene(self.data, camera=self._left,
                                scene_option=self._opt)
        depth = np.asarray(self._rend.render(), dtype=np.float32)
        self._rend.disable_depth_rendering()
        misses = (~np.isfinite(depth) | (depth <= 0.0)
                  | (depth >= self._far * 0.99))
        depth[misses] = 0.0
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)

        gray_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2GRAY)
        gray_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2GRAY)
        jpg = None
        if encode_jpeg:
            ok, buf = cv2.imencode(".jpg", rgb_l[:, :, ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpg = buf.tobytes()

        R = self.data.cam_xmat[self._left].reshape(3, 3)
        t = self.data.cam_xpos[self._left]
        R_opt, t_opt = optical((R, t))
        return gray_l, gray_r, depth_mm, R_opt, t_opt, jpg

    def close(self) -> None:
        self._rend.close()


__all__ = ["MujocoContractRenderer"]
