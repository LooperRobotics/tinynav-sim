# Vendored from parkour_mjlab @ train/orient-ablation (da1ac23):
#   deploy/pie/sim2sim/policy_runtime.py (contract import redirected to
#   pie_contract). Owns the GRU memory and validates the ONNX metadata
#   against pie_contract constants.
"""Strict ONNX runtime wrapper for Unitree-Go2-PIE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contract import (
    ACTION_SCALE,
    ACTUATED_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    DEPTH_HISTORY_SHAPE,
    JOINT_DAMPING,
    JOINT_STIFFNESS,
    NUM_ACTIONS,
    OBSERVATION_NAMES,
    OBS_DIM,
    PROPRIO_HISTORY_DIM,
)


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


def _metadata_floats(metadata: dict[str, str], key: str) -> np.ndarray:
    try:
        values = [float(item) for item in metadata[key].split(",") if item]
    except KeyError as exc:
        raise ValueError(f"ONNX metadata is missing {key!r}.") from exc
    return np.asarray(values, dtype=np.float32)


class PieOnnxPolicy:
    """ONNX policy wrapper that owns PIE's exported GRU state."""

    INPUT_SHAPES = {
        "proprio": (1, OBS_DIM),
        "proprio_history": (1, PROPRIO_HISTORY_DIM),
        "depth_history": (1, *DEPTH_HISTORY_SHAPE),
        "memory_h_in": (1, 1, 128),
    }
    OUTPUT_SHAPES = {
        "actions": (1, NUM_ACTIONS),
        "memory_h_out": (1, 1, 128),
    }

    def __init__(self, path: Path, provider: str = "auto") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to execute the exported PIE policy."
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
                "ONNX input contract does not match Unitree-Go2-PIE: "
                f"expected={self.INPUT_SHAPES}, actual={actual_inputs}."
            )
        if actual_outputs != self.OUTPUT_SHAPES:
            raise ValueError(
                "ONNX output contract does not match Unitree-Go2-PIE: "
                f"expected={self.OUTPUT_SHAPES}, actual={actual_outputs}."
            )

        joint_names = tuple(self.metadata.get("joint_names", "").split(","))
        if joint_names != ACTUATED_JOINT_NAMES:
            raise ValueError(
                "ONNX joint order does not match PIE: "
                f"expected={ACTUATED_JOINT_NAMES}, actual={joint_names}."
            )
        observation_names = tuple(self.metadata.get("observation_names", "").split(","))
        if observation_names != OBSERVATION_NAMES:
            raise ValueError(
                "ONNX observation order does not match PIE: "
                f"expected={OBSERVATION_NAMES}, actual={observation_names}."
            )
        command_names = tuple(self.metadata.get("command_names", "").split(","))
        if command_names != ("twist",):
            raise ValueError(f"Unexpected ONNX command metadata: {command_names}.")

        metadata_default = _metadata_floats(self.metadata, "default_joint_pos")
        if metadata_default.shape != DEFAULT_JOINT_POS.shape or not np.allclose(
            metadata_default, DEFAULT_JOINT_POS, atol=1.0e-3
        ):
            raise ValueError("ONNX default_joint_pos does not match PIE.")
        metadata_scale = _metadata_floats(self.metadata, "action_scale")
        # MJLab serializes a scalar JointPositionAction scale as one value,
        # while a per-joint configuration is serialized as twelve values.
        if metadata_scale.shape == (1,):
            metadata_scale = np.full_like(ACTION_SCALE, metadata_scale.item())
        if metadata_scale.shape != ACTION_SCALE.shape or not np.allclose(
            metadata_scale, ACTION_SCALE, atol=1.0e-3
        ):
            raise ValueError("ONNX action_scale does not match PIE.")
        metadata_stiffness = _metadata_floats(self.metadata, "joint_stiffness")
        if metadata_stiffness.shape != JOINT_STIFFNESS.shape or not np.allclose(
            metadata_stiffness, JOINT_STIFFNESS, atol=1.0e-3
        ):
            raise ValueError("ONNX joint_stiffness does not match PIE.")
        metadata_damping = _metadata_floats(self.metadata, "joint_damping")
        if metadata_damping.shape != JOINT_DAMPING.shape or not np.allclose(
            metadata_damping, JOINT_DAMPING, atol=1.0e-3
        ):
            raise ValueError("ONNX joint_damping does not match PIE.")

    def reset(self) -> None:
        self.memory_h = np.zeros((1, 1, 128), dtype=np.float32)

    def __call__(
        self,
        proprio: np.ndarray,
        proprio_history: np.ndarray,
        depth_history: np.ndarray,
    ) -> np.ndarray:
        inputs = {
            "proprio": np.asarray(proprio, dtype=np.float32).reshape(1, OBS_DIM),
            "proprio_history": np.asarray(proprio_history, dtype=np.float32).reshape(
                1, PROPRIO_HISTORY_DIM
            ),
            "depth_history": np.asarray(depth_history, dtype=np.float32).reshape(
                1, *DEPTH_HISTORY_SHAPE
            ),
            "memory_h_in": self.memory_h,
        }
        outputs = self.session.run(["actions", "memory_h_out"], inputs)
        action = np.asarray(outputs[0], dtype=np.float32)
        memory_h = np.asarray(outputs[1], dtype=np.float32)
        if action.shape != self.OUTPUT_SHAPES["actions"]:
            raise ValueError(f"Policy actions have invalid shape {action.shape}.")
        if memory_h.shape != self.OUTPUT_SHAPES["memory_h_out"]:
            raise ValueError(f"Policy memory has invalid shape {memory_h.shape}.")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(memory_h)):
            raise FloatingPointError("PIE policy produced NaN or infinity.")
        self.memory_h = memory_h
        return action.reshape(NUM_ACTIONS)


# go2_rl_gym (wty-yy) blind velocity actor: 45-D proprioception only, no
# depth and no GRU.  The deploy config in that repo is the reference for the
# scales, stance and PD gains; the joint order (FL,FR,RL,RR) matches PIE's.
GO2_RL_GYM_DEFAULT_JOINT_POS = np.asarray(
    (
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 1.0, -1.5,
        -0.1, 1.0, -1.5,
    ),
    dtype=np.float32,
)
GO2_RL_GYM_CMD_SCALE = np.asarray((2.0, 2.0, 0.25), dtype=np.float32)


class Go2RlGymOnnxPolicy:
    """Blind 45-D velocity actor exported from go2_rl_gym's MoE checkpoint.

    The runtime feeds the PIE 45-D vector (unscaled ang_vel, gravity, command,
    joint_pos - PIE_DEFAULT, joint_vel, last PIE action).  This class rescales
    it to that repo's actor contract and stores its own raw action for the
    last_action slot.  The returned action is expressed in the runtime's
    PIE-default/action-scale parameterization, so PieRuntime's
    `DEFAULT_JOINT_POS + ACTION_SCALE * action` yields the policy's own
    `GO2_RL_GYM_DEFAULT + 0.25 * action`.
    """

    needs_depth = False
    kp = 20.0
    kd = 0.5

    def __init__(self, path: Path, provider: str = "auto") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to execute the blind policy."
            ) from exc

        options = ort.SessionOptions()
        options.log_severity_level = 3
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
        inputs = {value.name: _fixed_shape(value.shape, "input")
                  for value in self.session.get_inputs()}
        if set(inputs) != {"obs"} or inputs["obs"] != (1, OBS_DIM):
            raise ValueError(
                "blind policy input contract must be a single (1, "
                f"{OBS_DIM}) 'obs'; got {inputs}."
            )
        outputs = {value.name: _fixed_shape(value.shape, "output")
                   for value in self.session.get_outputs()}
        if "actions" not in outputs or outputs["actions"] != (1, NUM_ACTIONS):
            raise ValueError(
                "blind policy must expose (1, 12) 'actions'; "
                f"got {outputs}."
            )
        self.reset()

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def reset(self) -> None:
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def __call__(
        self,
        proprio: np.ndarray,
        proprio_history: np.ndarray | None = None,
        depth_history: np.ndarray | None = None,
    ) -> np.ndarray:
        del proprio_history, depth_history
        pie = np.asarray(proprio, dtype=np.float32).reshape(OBS_DIM)
        observation = np.concatenate(
            (
                pie[0:3] * 0.25,                          # base ang vel
                pie[3:6],                                 # projected gravity
                pie[6:9] * GO2_RL_GYM_CMD_SCALE,          # command
                # qpos - GO2 default = (qpos - PIE default) + PIE - GO2
                pie[9:21] + DEFAULT_JOINT_POS
                - GO2_RL_GYM_DEFAULT_JOINT_POS,
                pie[21:33] * 0.05,                        # joint vel
                self._last_action,                        # raw policy action
            )
        ).astype(np.float32)
        action = self.session.run(
            ["actions"], {"obs": observation.reshape(1, -1)}
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(NUM_ACTIONS)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("blind policy produced NaN or infinity.")
        self._last_action = action
        # PieRuntime applies DEFAULT_JOINT_POS + ACTION_SCALE * returned; add
        # the constant stance offset so the actuator target is the go2_rl_gym
        # target: GO2_DEFAULT + 0.25 * action.
        return (
            (GO2_RL_GYM_DEFAULT_JOINT_POS - DEFAULT_JOINT_POS) / ACTION_SCALE
            + action
        ).astype(np.float32)


class PieBlindOnnxPolicy:
    """Blind flat velocity actor trained with mjlab on the PIE robot.

    Contract (mujoco/train/, parity-checked against sim/plant.py): a plain
    MLP over the exact 45-D PIE proprioception vector, position-actuator
    action default + 0.25*action, PIE gains stay in the plant model (no
    kp/kd override).  Needs no depth render and no GRU state.  The ONNX
    input must be named 'proprio' (the training export's 'obs' is renamed
    when the asset is promoted, so load_policy can route on it).
    """

    needs_depth = False

    def __init__(self, path: Path, provider: str = "auto") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to execute the blind policy."
            ) from exc

        options = ort.SessionOptions()
        options.log_severity_level = 3
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
        self._validate_contract()
        self.reset()

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _validate_contract(self) -> None:
        inputs = {v.name: _fixed_shape(v.shape, f"input {v.name!r}")
                  for v in self.session.get_inputs()}
        outputs = {v.name: _fixed_shape(v.shape, f"output {v.name!r}")
                   for v in self.session.get_outputs()}
        if inputs != {"proprio": (1, OBS_DIM)}:
            raise ValueError(
                "blind PIE policy must take a single (1, "
                f"{OBS_DIM}) 'proprio'; got {inputs}."
            )
        if outputs.get("actions") != (1, NUM_ACTIONS):
            raise ValueError(
                f"blind PIE policy must expose (1, {NUM_ACTIONS}) 'actions'; "
                f"got {outputs}."
            )
        joint_names = tuple(self.metadata.get("joint_names", "").split(","))
        if joint_names != ACTUATED_JOINT_NAMES:
            raise ValueError(
                "blind PIE policy joint order does not match the contract: "
                f"expected={ACTUATED_JOINT_NAMES}, actual={joint_names}."
            )
        observation_names = tuple(
            self.metadata.get("observation_names", "").split(",")
        )
        if observation_names != OBSERVATION_NAMES:
            raise ValueError(
                "blind PIE policy observation order does not match PIE: "
                f"expected={OBSERVATION_NAMES}, actual={observation_names}."
            )
        metadata_default = _metadata_floats(self.metadata, "default_joint_pos")
        if metadata_default.shape != DEFAULT_JOINT_POS.shape or not np.allclose(
            metadata_default, DEFAULT_JOINT_POS, atol=1.0e-3
        ):
            raise ValueError("blind PIE policy default_joint_pos mismatch.")
        metadata_scale = _metadata_floats(self.metadata, "action_scale")
        if metadata_scale.shape == (1,):
            metadata_scale = np.full_like(ACTION_SCALE, metadata_scale.item())
        if metadata_scale.shape != ACTION_SCALE.shape or not np.allclose(
            metadata_scale, ACTION_SCALE, atol=1.0e-3
        ):
            raise ValueError("blind PIE policy action_scale mismatch.")

    def reset(self) -> None:
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def __call__(
        self,
        proprio: np.ndarray,
        proprio_history: np.ndarray | None = None,
        depth_history: np.ndarray | None = None,
    ) -> np.ndarray:
        del proprio_history, depth_history
        observation = np.asarray(proprio, dtype=np.float32).reshape(1, OBS_DIM)
        action = self.session.run(["actions"], {"proprio": observation})[0]
        action = np.asarray(action, dtype=np.float32).reshape(NUM_ACTIONS)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("blind PIE policy produced NaN or infinity.")
        self._last_action = action
        return action


def load_policy(path: str | Path, provider: str = "auto"):
    """Pick the runtime wrapper from the ONNX input contract."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to load a policy.") from exc
    probe = ort.InferenceSession(
        str(Path(path)), providers=["CPUExecutionProvider"]
    )
    names = {value.name for value in probe.get_inputs()}
    if "depth_history" in names or "proprio_history" in names:
        return PieOnnxPolicy(path, provider)
    if "proprio" in names:
        return PieBlindOnnxPolicy(path, provider)
    if names == {"obs"}:
        shape = _fixed_shape(probe.get_inputs()[0].shape, "input obs")
        if len(shape) == 2 and shape[1] == OBS_DIM:
            return Go2RlGymOnnxPolicy(path, provider)
    raise ValueError(
        f"unrecognized policy contract in {path}: inputs={sorted(names)}"
    )


__all__ = [
    "Go2RlGymOnnxPolicy",
    "PieBlindOnnxPolicy",
    "PieOnnxPolicy",
    "load_policy",
    "resolve_onnx_path",
]
