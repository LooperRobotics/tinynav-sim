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

    def test_wheeled_v2_has_no_phase_and_a_brake(self):
        meta = wheeled_metadata() | {
            "contract": "go2w_depth_v2",
            "observation_names": "base_ang_vel,projected_gravity,command,"
                                 "joint_pos,joint_vel,actions",
            "wheel_brake_rad_s": "1.16279",
        }
        k = V.StudentContract(meta)
        self.assertEqual(k.OBSERVATION_TERM_DIMS, (3, 3, 3, 12, 16, 16))
        self.assertEqual((k.OBS_DIM, k.PROPRIO_HISTORY_DIM), (53, 530))
        self.assertAlmostEqual(k.wheel_brake_rad_s, 1.16279)
        obs = k.build_proprioception(
            np.zeros(3), (1.0, 0.0, 0.0, 0.0), np.array([0.5, 0.0, 0.0], np.float32),
            k.default_joint_pos, np.zeros(16), np.zeros(16), control_step=7)
        self.assertEqual(obs.shape, (53,))
        np.testing.assert_allclose(obs[6:9], [0.5, 0.0, 0.0])   # command, then joint_pos
        self.assertIsNone(V.StudentContract(wheeled_metadata()).wheel_brake_rad_s)

    def test_observation_terms_out_of_order_are_refused(self):
        meta = wheeled_metadata() | {
            "observation_names": "projected_gravity,base_ang_vel,command,"
                                 "joint_pos,joint_vel,actions"}
        with self.assertRaises(ValueError):
            V.StudentContract(meta)

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


def heading_metadata(contract: str = "go2_depth_v2_heading") -> dict[str, str]:
    return {"contract": contract, "joint_names": ",".join(LEGS),
            "default_joint_pos": ",".join(f"{v:.3f}" for v in C.DEFAULT_JOINT_POS),
            "action_scale": "0.25",
            "observation_names": ",".join(V.OBSERVATION_NAMES + (V.HEADING_TERM,))}


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


class TestHeading(unittest.TestCase):
    def test_heading_student_dims_and_envelope(self):
        k = V.contract_for(heading_metadata())
        self.assertTrue(k.heading)
        self.assertEqual(k.OBSERVATION_TERM_DIMS[-1], 1)
        self.assertEqual((k.OBS_DIM, k.PROPRIO_HISTORY_DIM), (48, 480))
        # trained on go2-blind's command ranges
        np.testing.assert_allclose(k.shape_command((2.0, -1.0, 2.0)), (1.5, -0.6, 1.5))
        obs = k.build_proprioception(np.zeros(3), (1, 0, 0, 0), np.zeros(3),
                                     k.default_joint_pos, np.zeros(12), np.zeros(12),
                                     0, heading_error=0.25)
        self.assertEqual(obs[-1], np.float32(0.25))
        with self.assertRaises(ValueError):
            k.build_proprioception(np.zeros(3), (1, 0, 0, 0), np.zeros(3),
                                   k.default_joint_pos, np.zeros(12), np.zeros(12), 0)
        self.assertIsInstance(k.HeadingError(), V.HeadingError)
        self.assertIsNone(V.StudentContract(wheeled_metadata()).HeadingError())

    def test_heading_contract_needs_the_term(self):
        meta = heading_metadata() | {"observation_names": ",".join(V.OBSERVATION_NAMES)}
        with self.assertRaises(ValueError):
            V.StudentContract(meta)
        with self.assertRaises(ValueError):
            V.StudentContract(heading_metadata("go2_depth_v2"))

    def test_wheeled_heading_student(self):
        meta = wheeled_metadata() | {
            "contract": "go2w_depth_v2_heading",
            "observation_names": "base_ang_vel,projected_gravity,command,"
                                 "joint_pos,joint_vel,actions,heading_error",
        }
        k = V.StudentContract(meta)
        self.assertEqual((k.OBS_DIM, k.PROPRIO_HISTORY_DIM), (54, 540))

    def test_heading_blind_export(self):
        meta = heading_metadata()
        del meta["contract"]
        k = V.contract_for(meta, ["obs"])
        self.assertIsInstance(k, V.BlindContract)
        self.assertEqual(k.OBS_DIM, 48)

    def test_heading_error_settles_integrates_and_clamps(self):
        h = V.HeadingError()
        # target = base yaw for the first two steps, whatever the command
        self.assertEqual(h(yaw_quat(0.3), 1.0), 0.0)
        self.assertEqual(h(yaw_quat(0.4), 1.0), 0.0)
        # then it integrates the yaw-rate command (rad/s) every 0.02 s step
        self.assertAlmostEqual(h(yaw_quat(0.4), 1.0), 0.02, places=6)
        self.assertAlmostEqual(h(yaw_quat(0.4), 1.0), 0.04, places=6)
        # held within 0.5 rad of the base yaw: a stalled turn saturates, and
        # the error unwinds as soon as the robot turns
        for _ in range(100):
            error = h(yaw_quat(0.4), 1.0)
        self.assertAlmostEqual(error, 0.5, places=6)
        self.assertAlmostEqual(h(yaw_quat(0.8), 0.0), 0.1, places=6)
        # wrapped across +-pi: turned 0.083 rad left past the target
        h = V.HeadingError()
        h(yaw_quat(3.1), 0.0)
        h(yaw_quat(3.1), 0.0)
        self.assertAlmostEqual(h(yaw_quat(-3.1), 0.0), 6.2 - 2 * np.pi, places=5)
        h.reset()
        self.assertEqual(h(yaw_quat(1.0), 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
