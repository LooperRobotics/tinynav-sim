"""MapWriter interface + v2-npy implementation.

V2NpyWriter produces exactly the layout tools/export_map_v2.py produces
(C++ reader: src/tinynav_cpp/src/mapping/map_v2.cpp), plus the semantic
sidecar from docs/plan-siglip2-map-builder.md:

  pose_timestamps.npy [N] i64
  pose_matrices.npy   [N,4,4] f64
  vlad_centres.npy    [C,d]
  vlad_descriptors.npy [N,d] f32
  feature_offsets.npy [N+1] i64
  feature_kpts.npy    [M,2] f32 / feature_descps.npy [M,D] f32 / feature_mask.npy [M] u1
  depth_images.npy    [N,H,W] u16 millimeters
  semantic_embeddings.npy [N,768] f32 (sidecar; zero row = missing)
  semantic_meta.json  {model, model_version, normalized, key:"ts_ns", created, builder_version}
  meta.json           informational, same fields as export_map_v2.py

All rows follow ascending keyframe timestamp; add_keyframe order does not
matter, finalize() sorts.
"""
from __future__ import annotations

import abc
import json
import os
import time

import numpy as np

SEMANTIC_MODEL = "google/siglip2-base-patch16-224"
SEMANTIC_MODEL_VERSION = "siglip2_base_p16_224_fp16"
BUILDER_VERSION = "0.1.0"
SEMANTIC_DIM = 768


class MapWriter(abc.ABC):
    """One map build. Call order is free except finalize() last."""

    @abc.abstractmethod
    def add_keyframe(self, ts_ns: int, pose_world_from_camera: np.ndarray) -> None:
        """pose: 4x4 f64-ish, camera-to-world."""

    @abc.abstractmethod
    def add_features(self, ts_ns: int, kpts: np.ndarray, descps: np.ndarray,
                     mask: np.ndarray) -> None:
        """SuperPoint output; kpts [N,2] px, descps [N,D], mask [N]."""

    @abc.abstractmethod
    def add_depth(self, ts_ns: int, depth_m: np.ndarray) -> None:
        """HxW f32 meters; writer stores u16 millimeters (0 = invalid)."""

    @abc.abstractmethod
    def add_vlad(self, ts_ns: int, descriptor: np.ndarray) -> None:
        """f32/f64 [d] per keyframe."""

    @abc.abstractmethod
    def set_vlad_centres(self, centres: np.ndarray) -> None:
        """[C,d] vocabulary, written once."""

    @abc.abstractmethod
    def add_semantic(self, ts_ns: int, embedding: np.ndarray) -> None:
        """Unit-norm f32 [768]; row follows the keyframe in the sidecar."""

    @abc.abstractmethod
    def add_aux(self, name: str, array: np.ndarray) -> None:
        """Plain .npy passthroughs (intrinsics/occupancy/sdf/path_*)."""

    @abc.abstractmethod
    def finalize(self) -> None:
        """Sort by timestamp, write everything, numpy-readback validate."""


class V2NpyWriter(MapWriter):
    def __init__(self, out_dir: str, semantic_model: str = SEMANTIC_MODEL,
                 semantic_model_version: str = SEMANTIC_MODEL_VERSION):
        self.out_dir = out_dir
        self.semantic_model = semantic_model
        self.semantic_model_version = semantic_model_version
        self._keyframes: dict[int, dict] = {}
        self._vlad_centres: np.ndarray | None = None
        self._aux: dict[str, np.ndarray] = {}

    def add_keyframe(self, ts_ns: int, pose_world_from_camera: np.ndarray) -> None:
        pose = np.asarray(pose_world_from_camera, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"pose for ts={ts_ns} must be 4x4, got {pose.shape}")
        self._keyframes[int(ts_ns)] = {"pose": pose}

    def _kf(self, ts_ns: int) -> dict:
        ts = int(ts_ns)
        if ts not in self._keyframes:
            raise KeyError(f"ts={ts} not added via add_keyframe")
        return self._keyframes[ts]

    def add_features(self, ts_ns: int, kpts: np.ndarray, descps: np.ndarray,
                     mask: np.ndarray) -> None:
        k = np.asarray(kpts)
        if k.ndim == 3:
            k = k[0]
        if k.ndim == 2 and k.shape[0] == 2 and k.shape[1] != 2:
            k = k.T
        d = np.asarray(descps)
        if d.ndim == 3:
            d = d[0]
        m = np.asarray(mask).reshape(-1)
        k, d, m = k.astype(np.float32), d.astype(np.float32), m.astype(np.uint8)
        if d.shape[0] != k.shape[0] or m.shape[0] != k.shape[0]:
            raise ValueError(
                f"features for ts={ts_ns} inconsistent: kpts {k.shape}, descps {d.shape}, mask {m.shape}")
        self._kf(ts_ns)["features"] = (k, d, m)

    def add_depth(self, ts_ns: int, depth_m: np.ndarray) -> None:
        dep = np.asarray(depth_m, dtype=np.float32)
        if dep.ndim != 2:
            raise ValueError(f"depth for ts={ts_ns} must be HxW, got {dep.shape}")
        self._kf(ts_ns)["depth"] = dep

    def add_vlad(self, ts_ns: int, descriptor: np.ndarray) -> None:
        self._kf(ts_ns)["vlad"] = np.asarray(descriptor).astype(np.float32).reshape(-1)

    def set_vlad_centres(self, centres: np.ndarray) -> None:
        self._vlad_centres = np.asarray(centres, dtype=np.float32)

    def add_semantic(self, ts_ns: int, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape != (SEMANTIC_DIM,):
            raise ValueError(f"semantic embedding for ts={ts_ns} must be [{SEMANTIC_DIM}], got {emb.shape}")
        self._kf(ts_ns)["semantic"] = emb

    def add_aux(self, name: str, array: np.ndarray) -> None:
        if not name.endswith(".npy"):
            name += ".npy"
        self._aux[name] = np.asarray(array)

    def finalize(self) -> None:
        timestamps = sorted(self._keyframes)
        n = len(timestamps)
        if n == 0:
            raise RuntimeError("no keyframes added")
        missing_depth = [t for t in timestamps if "depth" not in self._keyframes[t]]
        if missing_depth:
            raise RuntimeError(f"keyframes without depth: {missing_depth[:5]}...")
        depth_hw = self._keyframes[timestamps[0]]["depth"].shape
        for t in timestamps:
            if self._keyframes[t]["depth"].shape != depth_hw:
                raise RuntimeError(f"depth shape mismatch at ts={t}")

        pose_matrices = np.stack([self._keyframes[t]["pose"] for t in timestamps])
        vlad_rows = [self._keyframes[t].get("vlad") for t in timestamps]
        if any(v is None for v in vlad_rows):
            raise RuntimeError("vlad descriptors incomplete")
        vlad_descriptors = np.stack(vlad_rows)
        if self._vlad_centres is None:
            raise RuntimeError("set_vlad_centres never called")

        offsets, kpts_parts, desc_parts, mask_parts = [0], [], [], []
        for t in timestamps:
            feat = self._keyframes[t].get("features")
            if feat is None:
                offsets.append(offsets[-1])
                continue
            k, d, m = feat
            kpts_parts.append(k)
            desc_parts.append(d)
            mask_parts.append(m)
            offsets.append(offsets[-1] + k.shape[0])
        feature_kpts = np.concatenate(kpts_parts) if kpts_parts else np.zeros((0, 2), np.float32)
        feature_descps = np.concatenate(desc_parts) if desc_parts else np.zeros((0, 256), np.float32)
        feature_mask = np.concatenate(mask_parts) if mask_parts else np.zeros((0,), np.uint8)

        depth_images = np.clip(
            np.stack([self._keyframes[t]["depth"] for t in timestamps]).astype(np.float64) * 1000.0,
            0.0, 65535.0).round().astype(np.uint16)

        semantic = np.zeros((n, SEMANTIC_DIM), np.float32)
        n_sem = 0
        for i, t in enumerate(timestamps):
            emb = self._keyframes[t].get("semantic")
            if emb is not None:
                semantic[i] = emb
                n_sem += 1

        os.makedirs(self.out_dir, exist_ok=True)
        written: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

        def save(name: str, arr: np.ndarray) -> None:
            path = os.path.join(self.out_dir, name)
            np.save(path, arr)
            written[name] = (arr.shape, arr.dtype)

        save("pose_timestamps.npy", np.asarray(timestamps, dtype=np.int64))
        save("pose_matrices.npy", pose_matrices)
        save("vlad_centres.npy", self._vlad_centres)
        save("vlad_descriptors.npy", vlad_descriptors)
        save("feature_offsets.npy", np.asarray(offsets, dtype=np.int64))
        save("feature_kpts.npy", feature_kpts)
        save("feature_descps.npy", feature_descps)
        save("feature_mask.npy", feature_mask)
        save("depth_images.npy", depth_images)
        save("semantic_embeddings.npy", semantic)
        for name, arr in self._aux.items():
            save(name, arr)

        meta = {
            "format": "tinynav_map_v2",
            "version": 2,
            "depth_dtype": "u2_mm",
            "count": int(n),
            "depth_height": int(depth_hw[0]),
            "depth_width": int(depth_hw[1]),
            "desc_dim": int(feature_descps.shape[1]) if feature_descps.ndim == 2 else 0,
            "vlad_dim": int(self._vlad_centres.shape[1]),
            "vlad_centres": int(self._vlad_centres.shape[0]),
            "feature_rows": int(feature_kpts.shape[0]),
            "semantic": {
                "embeddings": n_sem,
                "dim": SEMANTIC_DIM,
                "file": "semantic_embeddings.npy",
                "meta_file": "semantic_meta.json",
                "row_order": "pose_timestamps",
                "missing_row": "zeros",
            },
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        semantic_meta = {
            "model": self.semantic_model,
            "model_version": self.semantic_model_version,
            "normalized": True,
            "key": "ts_ns",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "builder_version": BUILDER_VERSION,
        }
        with open(os.path.join(self.out_dir, "semantic_meta.json"), "w") as fh:
            json.dump(semantic_meta, fh, indent=2)

        self._readback_validate(written)
        print(f"map v2: {n} keyframes ({n_sem} semantic), {feature_kpts.shape[0]} feature rows, "
              f"vlad {vlad_descriptors.shape}, depth {depth_images.shape} -> {self.out_dir}")

    def _readback_validate(self, written: dict[str, tuple[tuple[int, ...], np.dtype]]) -> None:
        for name, (shape, dtype) in written.items():
            arr = np.load(os.path.join(self.out_dir, name))
            if arr.shape != shape or arr.dtype != dtype:
                raise RuntimeError(
                    f"readback mismatch {name}: wrote {shape}/{dtype}, read {arr.shape}/{arr.dtype}")
