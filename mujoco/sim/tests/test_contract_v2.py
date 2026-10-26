"""Depth-student contract unit tests (metadata parsing, observation layout).
Run from mujoco/:  python3 -m unittest sim.tests.test_contract_v2 -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from sim import contract as C  # noqa: E402
from sim import contract_v2 as V  # noqa: E402

LEGS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
        for part in ("hip", "thigh", "calf")]
WHEELED = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
           for part in ("hip", "thigh", "calf", "wheel")]


def wheeled_metadata() -> dict[str, str]:
    default = [{"hip": (-0.1 if j.startswith(("FL", "RL")) else 0.1),
                "thigh": 0.9, "calf": -1.8, "wheel": 0.0}[j.split("_")[1]]
               for j in WHEELED]
    return {
        "contract": "go2w_depth_v1",
        "joint_names": ",".join(WHEELED),
        "default_joint_pos": ",".join(f"{v:.3f}" for v in default),
        "action_scale": "0.25",
        "action_layout": ("joint_pos[position]*0.25:" + ",".join(LEGS)
                          + ";wheel_vel[velocity]*10:"
                          + ",".join(j for j in WHEELED if "wheel" in j)),
    }


class TestStudentContract(unittest.TestCase):
    def test_layout(self):
        groups = V.parse_action_layout(wheeled_metadata()["action_layout"])
        self.assertEqual([(g.kind, g.scale, len(g.joints)) for g in groups],
                         [("position", 0.25, 12), ("velocity", 10.0, 4)])

    def test_wheeled_dims(self):
        k = V.StudentContract(wheeled_metadata())
        self.assertEqual(k.OBSERVATION_TERM_DIMS, (3, 3, 3, 2, 12, 16, 16))
        self.assertEqual((k.OBS_DIM, k.PROPRIO_HISTORY_DIM, k.NUM_ACTIONS),
                         (55, 550, 16))

    def test_legged_without_layout(self):
        meta = {"contract": "go2_depth_v2", "joint_names": ",".join(LEGS),
                "default_joint_pos": ",".join(f"{v:.3f}" for v in C.DEFAULT_JOINT_POS),
                "action_scale": "0.25"}
        k = V.StudentContract(meta)
        self.assertEqual((k.OBS_DIM, k.NUM_ACTIONS), (47, 12))

    def test_joint_pos_drops_wheels_joint_vel_keeps_them(self):
        k = V.StudentContract(wheeled_metadata())
        pos = np.asarray(k.default_joint_pos, np.float64) + np.arange(16) * 0.01
        vel = np.arange(16, dtype=np.float64)
        obs = k.build_proprioception(np.zeros(3), (1, 0, 0, 0), np.zeros(3),
                                     pos, vel, np.zeros(16), 0)
        legs = [i for i, j in enumerate(WHEELED) if "wheel" not in j]
        np.testing.assert_allclose(obs[11:23], np.arange(16)[legs] * 0.01, atol=1e-6)
        np.testing.assert_allclose(obs[23:39], vel)
        np.testing.assert_allclose(obs[3:6], (0, 0, -1))   # projected gravity

    def test_command_shaping(self):
        k = V.StudentContract(wheeled_metadata())
        np.testing.assert_allclose(k.shape_command((2.0, -1.0, 0.0)), (1.0, -0.5, 0.0))
        np.testing.assert_allclose(k.shape_command((0.05, 0.05, 0.0)), 0.0)

    def test_history_is_term_major(self):
        h = V.TermHistory((2, 1), length=3)
        for i in range(3):
            h.append((i, i + 10, i + 100))
        np.testing.assert_allclose(h.array, (0, 10, 1, 11, 2, 12, 100, 101, 102))

    def test_unknown_contract_is_refused(self):
        with self.assertRaises(ValueError):
            V.contract_for({**wheeled_metadata(), "contract": "dog_v9"})

    def test_blind_export_is_recognised_by_its_signature(self):
        meta = {"joint_names": ",".join(LEGS),
                "default_joint_pos": ",".join(f"{v:.3f}" for v in C.DEFAULT_JOINT_POS),
                "action_scale": "0.25",
                "observation_names": ",".join(V.OBSERVATION_NAMES)}
        k = V.contract_for(meta, ["obs"])
        self.assertIsInstance(k, V.BlindContract)
        self.assertEqual((k.OBS_DIM, k.PROPRIO_HISTORY_DIM), (47, 470))
        np.testing.assert_allclose(k.shape_command((2.0, -1.0, 2.0)), (1.5, -0.6, 1.5))
        np.testing.assert_allclose(k.shape_command((0.1, 0.0, 0.0)), 0.0)   # <= 0.1
        with self.assertRaises(ValueError):    # same metadata, student-style inputs
            V.contract_for(meta, ["proprio", "proprio_history", "memory_h_in"])

    def test_exports_without_contract_are_refused(self):
        with self.assertRaises(ValueError):    # e.g. the retired PIE policies
            V.contract_for({})


if __name__ == "__main__":
    unittest.main()
