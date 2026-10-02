"""Map format v3: single-file SQLite map (docs/plan-siglip2-map-builder.md 阶段 3).

SQLiteWriter is the second MapWriter implementation; the builder writes
through the same interface as V2NpyWriter, so the format is a replaceable
detail. Layout:

  map.sqlite (WAL)
    meta(k TEXT PRIMARY KEY, v TEXT)          -- format=tinynav_map_v3;
                                              -- PRAGMA user_version=3;
                                              -- blobs_meta (JSON dtype/shape per blob)
                                              -- + per-blob keys blob.<name>.dtype/.shape/.json
    keyframes(ts INTEGER PRIMARY KEY, pose BLOB)             -- 16 f64 row-major 4x4
    arrays(ts INTEGER, name TEXT, dtype TEXT, shape TEXT, data BLOB,
          PRIMARY KEY(ts, name))              -- depth(<u2 mm)/feature_kpts/feature_descps/
                                              -- feature_mask/vlad_descriptor/semantic_embedding
                                              -- shape = JSON array text; dtype numpy-style
    images(ts INTEGER, kind TEXT, codec TEXT, data BLOB,
          PRIMARY KEY(ts, kind))              -- schema reserved; the builder stores no images
    blobs(name TEXT PRIMARY KEY, data BLOB)   -- vlad_centres/intrinsics/occupancy_grid/
                                              -- occupancy_meta/sdf_map/path_*/baseline/
                                              -- T_rgb_to_infra1/... + semantic_meta (JSON text)

load_map_v3(path) reads a v3 file back and returns numpy arrays as written.
finalize() runs the same readback discipline as V2NpyWriter: every array is
re-read through load_map_v3 and checked shape/dtype/bytes against the write.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np

from .writer import (  # noqa: F401  (re-exported for callers importing from mapio)
    BUILDER_VERSION,
    SEMANTIC_DIM,
    SEMANTIC_MODEL,
    SEMANTIC_MODEL_VERSION,
    MapWriter,
)

FORMAT_V3 = "tinynav_map_v3"
USER_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  k TEXT PRIMARY KEY,
  v TEXT
);
CREATE TABLE IF NOT EXISTS keyframes(
  ts INTEGER PRIMARY KEY,
  pose BLOB
);
CREATE TABLE IF NOT EXISTS arrays(
  ts INTEGER,
  name TEXT,
  dtype TEXT,
  shape TEXT,
  data BLOB,
  PRIMARY KEY(ts, name)
);
CREATE TABLE IF NOT EXISTS images(
  ts INTEGER,
  kind TEXT,
  codec TEXT,
  data BLOB,
  PRIMARY KEY(ts, kind)
);
CREATE TABLE IF NOT EXISTS blobs(
  name TEXT PRIMARY KEY,
  data BLOB
);
"""


def _dtype_str(arr: np.ndarray) -> str:
    if arr.dtype.byteorder == ">":
        raise ValueError(f"big-endian arrays not supported: {arr.dtype}")
    # numpy renders 1-byte dtypes as "|u1"; the schema and the C++ reader use "<"
    return arr.dtype.str.replace("|", "<", 1)


def _c(arr) -> np.ndarray:
    # ascontiguousarray promotes 0-d to 1-d — 0-d blobs (baseline scalar) must round-trip
    arr = np.asarray(arr)
    return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)


def _shape_str(shape) -> str:
    return json.dumps([int(d) for d in shape])


def _open_readonly(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def load_map_v3(path: str) -> dict:
    """Read a v3 map file back. numpy arrays come back exactly as written
    (dtype/shape from the schema columns, never guessed)."""
    conn = _open_readonly(path)
    try:
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        meta = {k: v for k, v in conn.execute("SELECT k, v FROM meta")}
        if user_version != USER_VERSION or meta.get("format") != FORMAT_V3:
            raise ValueError(
                f"{path}: not a map v3 (user_version={user_version}, "
                f"format={meta.get('format')!r})")

        rows = conn.execute("SELECT ts, pose FROM keyframes ORDER BY ts").fetchall()
        timestamps = np.array([r[0] for r in rows], dtype=np.int64)
        poses = np.frombuffer(b"".join(r[1] for r in rows), dtype="<f8").reshape(-1, 4, 4).copy()

        arrays = {}
        for ts, name, dtype, shape, data in conn.execute(
                "SELECT ts, name, dtype, shape, data FROM arrays"):
            arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(json.loads(shape))
            arrays[(int(ts), name)] = {"dtype": dtype, "shape": json.loads(shape), "array": arr}

        blobs_meta = json.loads(meta.get("blobs_meta", "{}"))
        blobs = {}
        for name, data in conn.execute("SELECT name, data FROM blobs"):
            bm = blobs_meta.get(name)
            if bm and bm.get("json"):
                blobs[name] = json.loads(bytes(data).decode())
            elif bm:
                blobs[name] = np.frombuffer(
                    data, dtype=np.dtype(bm["dtype"])).reshape(bm["shape"])
            else:
                blobs[name] = bytes(data)
        return {"meta": meta, "blobs_meta": blobs_meta, "timestamps": timestamps,
                "poses": poses, "arrays": arrays, "blobs": blobs}
    finally:
        conn.close()


class SQLiteWriter(MapWriter):
    def __init__(self, path: str, semantic_model: str = SEMANTIC_MODEL,
                 semantic_model_version: str = SEMANTIC_MODEL_VERSION):
        self.path = path
        self.semantic_model = semantic_model
        self.semantic_model_version = semantic_model_version
        self._keyframes: dict[int, dict] = {}
        self._vlad_centres: np.ndarray | None = None
        self._aux: dict[str, np.ndarray] = {}
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA user_version={USER_VERSION}")

    def add_keyframe(self, ts_ns: int, pose_world_from_camera: np.ndarray) -> None:
        pose = np.ascontiguousarray(pose_world_from_camera, dtype=np.float64)
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
        self._vlad_centres = np.ascontiguousarray(centres, dtype=np.float64)

    def add_semantic(self, ts_ns: int, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape != (SEMANTIC_DIM,):
            raise ValueError(f"semantic embedding for ts={ts_ns} must be [{SEMANTIC_DIM}], got {emb.shape}")
        self._kf(ts_ns)["semantic"] = emb

    def add_aux(self, name: str, array: np.ndarray) -> None:
        self._aux[name.removesuffix(".npy")] = _c(array)

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

        vlad_rows = [self._keyframes[t].get("vlad") for t in timestamps]
        if any(v is None for v in vlad_rows):
            raise RuntimeError("vlad descriptors incomplete")
        if self._vlad_centres is None:
            raise RuntimeError("set_vlad_centres never called")

        feature_rows = 0
        desc_dim = 0
        n_sem = 0

        conn = self._conn
        with conn:
            conn.execute("DELETE FROM arrays")
            conn.execute("DELETE FROM keyframes")
            conn.execute("DELETE FROM meta")
            conn.execute("DELETE FROM blobs")

            for t in timestamps:
                kf = self._keyframes[t]
                pose = np.asarray(kf["pose"], dtype="<f8")
                conn.execute("INSERT INTO keyframes(ts, pose) VALUES (?, ?)",
                             (t, pose.tobytes()))

                depth = np.clip(
                    np.asarray(kf["depth"], dtype=np.float64) * 1000.0,
                    0.0, 65535.0).round().astype("<u2")
                conn.execute(
                    "INSERT INTO arrays(ts, name, dtype, shape, data) VALUES (?,?,?,?,?)",
                    (t, "depth", "<u2", _shape_str(depth.shape), depth.tobytes()))

                feat = kf.get("features")
                if feat is not None:
                    k, d, m = feat
                    feature_rows += k.shape[0]
                    desc_dim = d.shape[1] if d.ndim == 2 else 0
                    for name, arr in (("feature_kpts", k), ("feature_descps", d),
                                      ("feature_mask", m)):
                        conn.execute(
                            "INSERT INTO arrays(ts, name, dtype, shape, data) VALUES (?,?,?,?,?)",
                            (t, name, _dtype_str(arr), _shape_str(arr.shape), arr.tobytes()))

                vlad = np.ascontiguousarray(kf["vlad"], dtype="<f4")
                conn.execute(
                    "INSERT INTO arrays(ts, name, dtype, shape, data) VALUES (?,?,?,?,?)",
                    (t, "vlad_descriptor", "<f4", _shape_str(vlad.shape), vlad.tobytes()))

                sem = kf.get("semantic")
                if sem is not None:
                    n_sem += 1
                    sem = np.ascontiguousarray(sem, dtype="<f4")
                    conn.execute(
                        "INSERT INTO arrays(ts, name, dtype, shape, data) VALUES (?,?,?,?,?)",
                        (t, "semantic_embedding", "<f4", _shape_str(sem.shape), sem.tobytes()))

            blobs_meta = {}
            conn.execute("INSERT INTO blobs(name, data) VALUES (?, ?)",
                         ("vlad_centres", self._vlad_centres.tobytes()))
            blobs_meta["vlad_centres"] = {
                "dtype": _dtype_str(self._vlad_centres),
                "shape": list(self._vlad_centres.shape)}
            for name, arr in self._aux.items():
                conn.execute("INSERT INTO blobs(name, data) VALUES (?, ?)",
                             (name, arr.tobytes()))
                blobs_meta[name] = {"dtype": _dtype_str(arr), "shape": list(arr.shape)}
            semantic_meta = {
                "model": self.semantic_model,
                "model_version": self.semantic_model_version,
                "normalized": True,
                "key": "ts_ns",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "builder_version": BUILDER_VERSION,
            }
            conn.execute("INSERT INTO blobs(name, data) VALUES (?, ?)",
                         ("semantic_meta", json.dumps(semantic_meta).encode()))
            blobs_meta["semantic_meta"] = {"json": True}

            meta = {
                "format": FORMAT_V3,
                "version": USER_VERSION,
                "depth_dtype": "u2_mm",
                "count": int(n),
                "depth_height": int(depth_hw[0]),
                "depth_width": int(depth_hw[1]),
                "desc_dim": int(desc_dim),
                "vlad_dim": int(self._vlad_centres.shape[1]),
                "vlad_centres": int(self._vlad_centres.shape[0]),
                "feature_rows": int(feature_rows),
                "semantic": json.dumps({"embeddings": n_sem, "dim": SEMANTIC_DIM,
                                        "key": "ts_ns", "missing_row": "absent"}),
                "blobs_meta": json.dumps(blobs_meta),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "builder_version": BUILDER_VERSION,
            }
            for bm_name, bm in blobs_meta.items():
                meta[f"blob.{bm_name}.json"] = "1" if bm.get("json") else "0"
                if not bm.get("json"):
                    meta[f"blob.{bm_name}.dtype"] = bm["dtype"]
                    meta[f"blob.{bm_name}.shape"] = json.dumps(bm["shape"])
            conn.executemany("INSERT INTO meta(k, v) VALUES (?, ?)", list(meta.items()))
        # checkpoint so the file stands alone (readers may open it read-only
        # without the -wal sidecar)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        self._conn = None

        self._readback_validate(timestamps)
        print(f"map v3: {n} keyframes ({n_sem} semantic), {feature_rows} feature rows, "
              f"vlad [{n},{meta['vlad_dim']}], depth [{n},{depth_hw[0]},{depth_hw[1]}] -> {self.path}")

    def _readback_validate(self, timestamps: list[int]) -> None:
        got = load_map_v3(self.path)
        if not np.array_equal(got["timestamps"], np.asarray(timestamps, dtype=np.int64)):
            raise RuntimeError("readback mismatch: keyframe timestamps")
        for i, t in enumerate(timestamps):
            kf = self._keyframes[t]
            if not np.array_equal(got["poses"][i], np.asarray(kf["pose"], dtype=np.float64)):
                raise RuntimeError(f"readback mismatch: pose ts={t}")
            want_depth = np.clip(
                np.asarray(kf["depth"], dtype=np.float64) * 1000.0,
                0.0, 65535.0).round().astype(np.uint16)
            row = got["arrays"].get((t, "depth"))
            if row is None or row["dtype"] != "<u2" or not np.array_equal(row["array"], want_depth):
                raise RuntimeError(f"readback mismatch: depth ts={t}")
            want_vlad = np.ascontiguousarray(kf["vlad"], dtype=np.float32)
            row = got["arrays"].get((t, "vlad_descriptor"))
            if row is None or row["dtype"] != "<f4" or not np.array_equal(row["array"], want_vlad):
                raise RuntimeError(f"readback mismatch: vlad ts={t}")
            feat = kf.get("features")
            if feat is not None:
                for name, want in zip(("feature_kpts", "feature_descps", "feature_mask"), feat):
                    row = got["arrays"].get((t, name))
                    if (row is None or np.dtype(row["dtype"]) != np.dtype(want.dtype)
                            or row["array"].shape != want.shape
                            or not np.array_equal(row["array"], want)):
                        raise RuntimeError(f"readback mismatch: {name} ts={t}")
            sem = kf.get("semantic")
            if sem is not None:
                row = got["arrays"].get((t, "semantic_embedding"))
                if row is None or not np.array_equal(row["array"], np.asarray(sem, np.float32)):
                    raise RuntimeError(f"readback mismatch: semantic ts={t}")
        centres = got["blobs"].get("vlad_centres")
        if centres is None or not np.array_equal(centres, self._vlad_centres):
            raise RuntimeError("readback mismatch: vlad_centres")
        for name, arr in self._aux.items():
            if not np.array_equal(got["blobs"].get(name), arr):
                raise RuntimeError(f"readback mismatch: blob {name}")
