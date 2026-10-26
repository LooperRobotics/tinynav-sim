# Derived from parkour_mjlab @ train/orient-ablation (da1ac23):
#   deploy/pie/sim2sim/policy_runtime.py. Owns the GRU memory and validates
#   the ONNX I/O against its depth-student contract (sim/contract_v2.py).
"""Strict ONNX runtime wrapper for the depth-student policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contract import DEPTH_HISTORY_SHAPE
from .contract_v2 import contract_for


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def resolve_onnx_path(checkpoint_file: str | Path) -> tuple[Path, Path | None]:
    """Resolve policy.onnx, optionally from the newest model_N.pt alias."""
    requested = _resolve_path(checkpoint_file)
    checkpoint_path: Path | None
    if requested.suffix.lower() == ".onnx":
        onnx_path = requested
        checkpoint_path = None
    elif requested.suffix.lower() == ".pt":
        checkpoint_path = requested
        onnx_path = requested.with_name("policy.onnx")
    else:
        raise ValueError(
            "--checkpoint-file must point to policy.onnx or model_N.pt; "
            f"got {requested}."
        )

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            requested_step = int(checkpoint_path.stem.removeprefix("model_"))
        except ValueError as exc:
            raise ValueError(
                "A .pt alias must be named model_N.pt; otherwise pass ONNX directly."
            ) from exc
        available_steps = []
        for candidate in checkpoint_path.parent.glob("model_*.pt"):
            try:
                available_steps.append(int(candidate.stem.removeprefix("model_")))
            except ValueError:
                continue
        if available_steps and requested_step != max(available_steps):
            raise ValueError(
                f"{checkpoint_path.name} is not the newest checkpoint in its run. "
                "policy.onnx is overwritten on save, so pass an explicitly exported "
                "ONNX for older checkpoints."
            )

    if not onnx_path.is_file():
        suffix = (
            " The sibling policy.onnx export is required." if checkpoint_path else ""
        )
        raise FileNotFoundError(f"ONNX policy not found: {onnx_path}.{suffix}")
    return onnx_path, checkpoint_path


def _fixed_shape(value: Sequence[Any], name: str) -> tuple[int, ...]:
    if not all(isinstance(dim, int) and dim > 0 for dim in value):
        raise ValueError(f"ONNX {name} must have a fixed shape, got {value}.")
    return tuple(int(dim) for dim in value)


class PieOnnxPolicy:
    """ONNX policy wrapper that owns the exported GRU state.

    Runs the depth-student exports (sim/contract_v2.py: proprio with a gait
    phase, any robot's joints and action layout; the I/O names are the PIE
    export's, hence the class name); `self.contract` is the contract object
    the runtime builds observations and actuator targets with. A policy
    without a depth_history input is blind: `uses_depth` is False and the
    runtime skips the depth camera. Blind exports (BlindContract: one `obs`
    input = the proprio history, `actions` out, no memory) run here too. The
    runtime checks joints, default pose and PD gains against the robot it
    drives.
    """

    def __init__(self, path: Path, provider: str = "auto") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to execute the exported policy."
            ) from exc

        options = ort.SessionOptions()
        options.log_severity_level = 3
        # splatsense local change: at 50 Hz the default intra-op pool
        # (= all cores, spin-waiting between the ~1 ms runs) burns ~14
        # cores on the host. This 0.7M-param policy is single-thread
        # bound anyway; measure before ever changing these.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        available = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA provider requested but unavailable; "
                    f"available providers: {available}."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.path = Path(path)
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=providers
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        input_names = [value.name for value in self.session.get_inputs()]
        self.contract = contract_for(self.metadata, input_names)
        self.uses_depth = "depth_history" in input_names
        # Blind exports: the history is the only input and there is no GRU.
        self.recurrent = "memory_h_in" in input_names
        if input_names == ["obs"]:
            self.INPUT_SHAPES = {"obs": (1, self.contract.PROPRIO_HISTORY_DIM)}
        else:
            self.INPUT_SHAPES = {
                "proprio": (1, self.contract.OBS_DIM),
                "proprio_history": (1, self.contract.PROPRIO_HISTORY_DIM),
                "memory_h_in": (1, 1, 128),
            }
        if self.uses_depth:
            self.INPUT_SHAPES["depth_history"] = (1, *DEPTH_HISTORY_SHAPE)
        self.OUTPUT_SHAPES = {"actions": (1, self.contract.NUM_ACTIONS)}
        if self.recurrent:
            self.OUTPUT_SHAPES["memory_h_out"] = (1, 1, 128)
        self._validate_contract()
        self.reset()

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _validate_contract(self) -> None:
        actual_inputs = {
            value.name: _fixed_shape(value.shape, f"input {value.name!r}")
            for value in self.session.get_inputs()
        }
        actual_outputs = {
            value.name: _fixed_shape(value.shape, f"output {value.name!r}")
            for value in self.session.get_outputs()
        }
        if actual_inputs != self.INPUT_SHAPES:
            raise ValueError(
                f"ONNX input contract does not match {self.contract.name}: "
                f"expected={self.INPUT_SHAPES}, actual={actual_inputs}."
            )
        if actual_outputs != self.OUTPUT_SHAPES:
            raise ValueError(
                f"ONNX output contract does not match {self.contract.name}: "
                f"expected={self.OUTPUT_SHAPES}, actual={actual_outputs}."
            )

        observation_names = tuple(self.metadata.get("observation_names", "").split(","))
        if observation_names != self.contract.OBSERVATION_NAMES:
            raise ValueError(
                "ONNX observation order does not match its contract: "
                f"expected={self.contract.OBSERVATION_NAMES}, "
                f"actual={observation_names}."
            )
        command_names = tuple(self.metadata.get("command_names", "").split(","))
        if command_names != ("twist",):
            raise ValueError(f"Unexpected ONNX command metadata: {command_names}.")

    def reset(self) -> None:
        self.memory_h = np.zeros((1, 1, 128), dtype=np.float32)

    def __call__(
        self,
        proprio: np.ndarray,
        proprio_history: np.ndarray,
        depth_history: np.ndarray,
    ) -> np.ndarray:
        history = np.asarray(proprio_history, dtype=np.float32).reshape(
            1, self.contract.PROPRIO_HISTORY_DIM
        )
        if not self.recurrent:
            action = np.asarray(self.session.run(["actions"], {"obs": history})[0],
                                dtype=np.float32)
            if action.shape != self.OUTPUT_SHAPES["actions"] or not np.all(
                    np.isfinite(action)):
                raise FloatingPointError(f"Blind policy output invalid: {action.shape}.")
            return action.reshape(self.contract.NUM_ACTIONS)
        inputs = {
            "proprio": np.asarray(proprio, dtype=np.float32).reshape(
                1, self.contract.OBS_DIM
            ),
            "proprio_history": history,
            "memory_h_in": self.memory_h,
        }
        if self.uses_depth:
            inputs["depth_history"] = np.asarray(
                depth_history, dtype=np.float32).reshape(1, *DEPTH_HISTORY_SHAPE)
        outputs = self.session.run(["actions", "memory_h_out"], inputs)
        action = np.asarray(outputs[0], dtype=np.float32)
        memory_h = np.asarray(outputs[1], dtype=np.float32)
        if action.shape != self.OUTPUT_SHAPES["actions"]:
            raise ValueError(f"Policy actions have invalid shape {action.shape}.")
        if memory_h.shape != self.OUTPUT_SHAPES["memory_h_out"]:
            raise ValueError(f"Policy memory has invalid shape {memory_h.shape}.")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(memory_h)):
            raise FloatingPointError("Policy produced NaN or infinity.")
        self.memory_h = memory_h
        return action.reshape(self.contract.NUM_ACTIONS)


__all__ = ["PieOnnxPolicy", "resolve_onnx_path"]
