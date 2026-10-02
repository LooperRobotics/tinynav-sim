#!/usr/bin/env python3
"""Builder node: consumes /slam keyframe topics, writes map v2 + SigLIP2 sidecar.

Semantics mirror reference/tinynav/core/build_map_node.py::BuildMapNode
(process / detect_loop_closure / maybe_run_global_refinement / save_mapping)
with these deliberate deviations (docs/plan-siglip2-map-builder.md):
- per-keyframe data kept in memory (depth/features/patch tokens/embeddings);
  no v1 shelve/VideoDB. Scale limit: sim/testing maps, not yishang-scale.
- output via tools/mapio.MapWriter (V2NpyWriter)
- semantic embeddings from the SigLIP2 plan (unit norm baked in)
- no TF/marker/pointcloud publishing
"""
from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from tinynav.core.build_map_node import (
    check_global_frames_ratio,
    find_loop,
    generate_occupancy_map,
    solve_pose_graph,
)
from tinynav.core.math_utils import estimate_pose, msg2np, tf2np
from tinynav.core.models_trt import Dinov2TRT, LightGlueTRT, SuperPointTRT
from tinynav.core.path_climb import compute_path_climb, n_climbing
from tinynav.core.path_speed import compute_path_speed
from tinynav.core.vlad import compute_vlad, train_vocabulary_streaming
from tf2_ros import TransformBroadcaster  # noqa: F401  (kept: reference parity)

log = logging.getLogger("build_map_siglip2")

DEFAULT_SIGLIP2_PLAN = ("/workspace/dm/model_test/siglip_cmp/siglip2/"
                        "siglip2_base_p16_224_image_fp16_x86_64.plan")


class BuilderNode(Node):
    def __init__(self, out_dir: str,
                 siglip2_plan: str = DEFAULT_SIGLIP2_PLAN,
                 global_frames_ratio: float = 1.1,
                 color_timeout_warn: int = 20):
        super().__init__("build_map_siglip2")
        if global_frames_ratio < 1.0:
            raise ValueError(f"global_frames_ratio must be >= 1.0, got {global_frames_ratio}")
        self.global_frames_ratio = global_frames_ratio
        self._global_prev_num_frames = 0
        self.out_dir = out_dir

        sys_path_tools = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if sys_path_tools not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path_tools)
        from mapio.writer import V2NpyWriter
        from build_map.siglip2_engine import Siglip2ImageEngine
        self.writer = V2NpyWriter(out_dir)
        self.siglip2 = Siglip2ImageEngine(siglip2_plan)

        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()

        self.bridge = CvBridge()
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.rgb_image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        # color watchdog: N keyframes without a single color frame -> WARN loud
        # (reference 1.4 lesson: silent color loss produced maps with no RGB)
        self._keyframes_seen = 0
        self._color_seen = 0
        self._color_timeout_warn = color_timeout_warn
        self.ts = ApproximateTimeSynchronizer(
            [self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub,
             self.rgb_image_sub], 200, 0.02)
        self.ts.registerCallback(self.keyframe_callback)

        self.tf_sub = Subscriber(self, TFMessage, "/tf")
        self.tf_sub.registerCallback(self.tf_callback)
        self.tf_static_sub = Subscriber(self, TFMessage, "/tf_static")
        self.tf_static_sub.registerCallback(self.tf_callback)

        self.K = None
        self.baseline = None
        self.odom: dict[int, np.ndarray] = {}
        self.pose_graph_used_pose: dict[int, np.ndarray] = {}
        self.relative_pose_constraint: list = []
        self.last_keyframe_timestamp = None

        self.depths: dict[int, np.ndarray] = {}
        self.features: dict[int, dict] = {}
        self.embeddings: dict[int, np.ndarray] = {}
        self.semantic_embeddings: dict[int, np.ndarray] = {}
        self.patch_tokens: dict[int, np.ndarray] = {}

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.T_rgb_to_infra1 = None
        self.rgb_camera_K = None
        self.rgb_camera_info_sub = Subscriber(self, CameraInfo, "/camera/camera/color/camera_info")
        self.rgb_camera_info_sub.registerCallback(self.rgb_camera_info_callback)

        self._save_completed = False
        os.makedirs(out_dir, exist_ok=True)

    # ---- reference-parity callbacks ----

    def rgb_camera_info_callback(self, msg: CameraInfo):
        if self.rgb_camera_K is None:
            self.rgb_camera_K = np.array(msg.k).reshape(3, 3)

    def info_callback(self, msg: CameraInfo):
        if self.K is None:
            log.info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)

    def tf_callback(self, msg: TFMessage):
        T_infra1_to_link = None
        T_infra1_optical_to_infra1 = None
        T_rgb_to_link = None
        T_rgb_optical_to_rgb = None
        for t in msg.transforms:
            frame_id, child_frame_id, T = tf2np(t)
            if frame_id == "camera_link" and child_frame_id == "camera_infra1_frame":
                T_infra1_to_link = T
            if frame_id == "camera_infra1_frame" and child_frame_id == "camera_infra1_optical_frame":
                T_infra1_optical_to_infra1 = T
            if frame_id == "camera_color_frame" and child_frame_id == "camera_color_optical_frame":
                T_rgb_optical_to_rgb = T
            if frame_id == "camera_link" and child_frame_id == "camera_color_frame":
                T_rgb_to_link = T
            if frame_id == "cam_left" and child_frame_id == "cam_rgb":
                self.T_rgb_to_infra1 = T
        if (T_infra1_optical_to_infra1 is not None and T_rgb_optical_to_rgb is not None
                and T_infra1_to_link is not None and T_rgb_to_link is not None):
            self.T_rgb_to_infra1 = (np.linalg.inv(T_infra1_optical_to_infra1)
                                    @ np.linalg.inv(T_infra1_to_link)
                                    @ T_rgb_to_link @ T_rgb_optical_to_rgb)

    def keyframe_callback(self, keyframe_image_msg, keyframe_odom_msg, depth_msg, rgb_image_msg):
        if self.K is None:
            return
        self.process(keyframe_image_msg, keyframe_odom_msg, depth_msg, rgb_image_msg)
        self._keyframes_seen += 1
        if self._keyframes_seen >= self._color_timeout_warn and self._color_seen == 0:
            log.warning(
                f"{self._keyframes_seen} keyframes processed but zero color frames "
                "matched — color topic mismatch? semantic embeddings will be empty "
                "(reference 1.4 silent-color-loss lesson)")

    # ---- core pipeline (reference semantics) ----

    def process(self, keyframe_image_msg, keyframe_odom_msg, depth_msg, rgb_image_msg):
        keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(
            keyframe_image_msg.header.stamp.nanosec)
        keyframe_odom_timestamp = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(
            keyframe_odom_msg.header.stamp.nanosec)
        keyframe_depth_timestamp = int(depth_msg.header.stamp.sec * 1e9) + int(
            depth_msg.header.stamp.nanosec)
        if keyframe_image_timestamp != keyframe_odom_timestamp or keyframe_image_timestamp != keyframe_depth_timestamp:
            log.error(f"Keyframe timestamp mismatch: {keyframe_image_timestamp} != {keyframe_odom_timestamp} != {keyframe_depth_timestamp}")

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        odom, _ = msg2np(keyframe_odom_msg)
        infra1_image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding="bgr8")
        self._color_seen += 1

        self.depths[keyframe_image_timestamp] = depth

        embedding, patch_tokens = asyncio.run(
            self.dinov2_model.infer_global_and_patch_tokens(infra1_image))
        embedding = embedding / np.linalg.norm(embedding)
        self.embeddings[keyframe_image_timestamp] = embedding
        self.patch_tokens[keyframe_image_timestamp] = patch_tokens

        self.semantic_embeddings[keyframe_image_timestamp] = self.siglip2.embed(rgb_image)

        features = asyncio.run(self.super_point_extractor.infer(infra1_image))
        self.features[keyframe_image_timestamp] = features

        if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
            self.odom[keyframe_image_timestamp] = odom
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
        else:
            last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
            T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
            self.relative_pose_constraint.append(
                (keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.odom[keyframe_image_timestamp] = odom
            self.detect_loop_closure(keyframe_image_timestamp)

        self.maybe_run_global_refinement()
        self.last_keyframe_timestamp = keyframe_image_timestamp

    def detect_loop_closure(self, timestamp: int) -> None:
        target_embedding = self.embeddings[timestamp]
        valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
        valid_embeddings = np.array([self.embeddings[t] for t in valid_timestamp])
        idx_to_timestamp = {i: t for i, t in enumerate(valid_timestamp)}

        loop_list = find_loop(target_embedding, valid_embeddings,
                              self.loop_similarity_threshold, self.loop_top_k)
        for idx, _similarity in loop_list:
            prev_timestamp = idx_to_timestamp[idx]
            curr_timestamp = timestamp
            prev_features = self.features[prev_timestamp]
            curr_features = self.features[curr_timestamp]
            prev_matched_keypoints, curr_matched_keypoints, _matches = self.match_keypoints(
                prev_features, curr_features)
            success, T_prev_curr, _, _, inliers = estimate_pose(
                prev_matched_keypoints, curr_matched_keypoints,
                self.depths[curr_timestamp], self.K)
            if success and len(inliers) >= 100:
                self.relative_pose_constraint.append(
                    (curr_timestamp, prev_timestamp, T_prev_curr))
                log.info(f"Added loop relative pose constraint: {curr_timestamp} -> {prev_timestamp}")

    def maybe_run_global_refinement(self) -> None:
        num_frames = len(self.pose_graph_used_pose)
        if not check_global_frames_ratio(num_frames, self._global_prev_num_frames,
                                         self.global_frames_ratio):
            return
        try:
            self.pose_graph_used_pose = solve_pose_graph(
                self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num=5)
        except Exception as e:
            log.warning(f"online pose graph solve failed, continuing unoptimized: {e}")
        self._global_prev_num_frames = num_frames

    def match_keypoints(self, feats0, feats1,
                        image_shape=np.array([848, 480], dtype=np.int64)):
        match_result = asyncio.run(self.light_glue_matcher.infer(
            feats0["kpts"], feats1["kpts"], feats0["descps"], feats1["descps"],
            feats0["mask"], feats1["mask"], image_shape, image_shape))
        match_indices = match_result["match_indices"][0]
        valid_mask = match_indices != -1
        keypoints0 = feats0["kpts"][0][valid_mask]
        keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
        matches = []
        for i, index in enumerate(match_indices):
            if index != -1:
                matches.append([i, index])
        return keypoints0, keypoints1, np.array(matches, dtype=np.int64)

    # ---- save ----

    def save_mapping(self):
        if self._save_completed:
            log.info("Mapping data already saved, skipping duplicate save")
            return
        if self.K is None:
            log.info("No camera intrinsics available, skipping save")
            return
        log.info("Saving mapping data...")
        n_keyframes = len(self.pose_graph_used_pose)

        try:
            self.pose_graph_used_pose = solve_pose_graph(
                self.pose_graph_used_pose, self.relative_pose_constraint)
        except Exception as e:
            log.warning(f"final pose graph solve failed, saving odom poses: {e}")

        timestamps = list(self.pose_graph_used_pose.keys())

        for t in timestamps:
            self.writer.add_keyframe(t, self.pose_graph_used_pose[t])
            self.writer.add_depth(t, self.depths[t])
            feat = self.features.get(t)
            if feat is not None:
                self.writer.add_features(t, feat["kpts"], feat["descps"], feat["mask"])
            if t in self.semantic_embeddings:
                self.writer.add_semantic(t, self.semantic_embeddings[t])

        try:
            path_speed = compute_path_speed(self.pose_graph_used_pose)
            self.writer.add_aux("path_speed.npy", path_speed)
        except Exception as e:
            log.error(f"Failed to compute path_speed: {e}")
        try:
            path_climb = compute_path_climb(self.pose_graph_used_pose)
            self.writer.add_aux("path_climb.npy", path_climb)
            log.info(f"path_climb: {n_climbing(path_climb)}/{len(path_climb)} samples climbing")
        except Exception as e:
            log.error(f"Failed to compute path_climb: {e}")

        self.writer.add_aux("intrinsics.npy", self.K)
        if self.baseline is not None:
            self.writer.add_aux("baseline.npy", np.asarray(self.baseline))
        if self.T_rgb_to_infra1 is not None:
            self.writer.add_aux("T_rgb_to_infra1.npy", self.T_rgb_to_infra1)
        if self.rgb_camera_K is not None:
            self.writer.add_aux("rgb_camera_intrinsics.npy", self.rgb_camera_K)

        log.info("Training VLAD vocabulary ...")
        vlad_train_rng = np.random.default_rng(42)

        def vlad_batch_iterator():
            order = vlad_train_rng.permutation(timestamps)
            for timestamp in order:
                yield self.patch_tokens[int(timestamp)]

        vlad_centres = train_vocabulary_streaming(vlad_batch_iterator)
        self.writer.set_vlad_centres(vlad_centres)
        for timestamp in timestamps:
            self.writer.add_vlad(
                timestamp, compute_vlad(self.patch_tokens[timestamp], vlad_centres))

        log.info("Generating occupancy map ...")
        occupancy_resolution = 0.1
        occupancy_step = 10
        occupancy_grid, occupancy_origin, _occ_img, sdf_map = generate_occupancy_map(
            self.pose_graph_used_pose, _DepthAdapter(self.depths), self.K,
            self.baseline, occupancy_resolution, occupancy_step)
        occupancy_meta = np.array(
            [occupancy_origin[0], occupancy_origin[1], occupancy_origin[2],
             occupancy_resolution], dtype=np.float32)
        self.writer.add_aux("occupancy_grid.npy", occupancy_grid)
        self.writer.add_aux("occupancy_meta.npy", occupancy_meta)
        self.writer.add_aux("sdf_map.npy", sdf_map)

        self.writer.finalize()
        self._save_completed = True
        log.info(f"Full mapping data saved ({n_keyframes} keyframes) -> {self.out_dir}")

    def destroy_node(self):
        try:
            self.save_mapping()
        except Exception as e:
            log.error(f"save during shutdown failed: {e}")
        super().destroy_node()


class _DepthAdapter:
    """generate_occupancy_map duck-type: only depth is consumed."""

    def __init__(self, depths: dict[int, np.ndarray]):
        self.depths = depths

    def get_depth_embedding_features_images(self, ts: int):
        return self.depths[ts], None, None, lambda: None, lambda: None
