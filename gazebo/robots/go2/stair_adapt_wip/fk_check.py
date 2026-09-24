#!/usr/bin/env python3
"""Offline unit checks for the stair-adaptation changes in go2_controller.py.

No ROS and no sim: rclpy/geometry_msgs/std_msgs are stubbed so the module
imports cleanly. Run inside the tinynav container:

    python3 gazebo/robots/go2/fk_check.py

Checks:
  T1  FK o IK round trip on a reachable grid, all four legs (< 1e-6 m)
  T2  pitch sign: +pitch_cmd drops front / raises rear foot targets (nose-up)
  T3  zero swing bias reproduces the legacy swing trajectory exactly
  T4  estimator on flat nominal angles: legacy behavior, climbing False
  T5  synthetic stair: front support high + blocked swing escalates the
      bias, pitches nose-up, flags climbing
  T6  bias decays after the evidence window
"""
import math
import os
import sys
import types


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub("rclpy")
_stub("rclpy.node", Node=type("Node", (), {}))
_stub("rclpy.time", Time=type("Time", (), {"seconds_nanoseconds": lambda self: (0, 0)}))
_stub("geometry_msgs")
_stub("geometry_msgs.msg", Twist=type("Twist", (), {}))
_stub("sensor_msgs")
_stub("sensor_msgs.msg", JointState=type("JointState", (), {"position": []}))
_stub("std_msgs")
_stub("std_msgs.msg", Float64MultiArray=type("Float64MultiArray", (), {}))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go2_controller as gc  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


ik = gc.InverseKinematics([0.3762, 0.0935], [0.0, 0.0955, 0.213, 0.213])

# nominal stance xy per leg (from the node's default_stance construction)
DX, DY = 0.3762 * 0.5, 0.3762 * 0.5 * 0 + (0.0935 * 0.5 + 0.0955)
NOMINAL_XY = [(DX + 0.02, -DY), (DX + 0.02, +DY), (-DX, -DY), (-DX, +DY)]

default_stance = gc.np.array([
    [DX + 0.02, DX + 0.02, -DX, -DX],
    [-DY, DY, -DY, DY],
    [0, 0, 0, 0]])

# ---------------------------------------------------------------- T1: round trip
worst = 0.0
count = 0
for leg, (x0, y0) in enumerate(NOMINAL_XY):
    for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for dz in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
            x, y, z = x0 + dx, y0, -0.25 + dz
            if x * x + y * y < ik.l2 ** 2 + 1e-4:
                continue
            target = gc.np.array([[x, x, x, x], [y, y, y, y], [z, z, z, z]])
            angles = ik.inverse_kinematics(target, 0, 0, 0, 0, 0, 0)
            got = ik.foot_position_body(angles[leg * 3:leg * 3 + 3], leg)
            err = max(abs(got[0] - x), abs(got[1] - y), abs(got[2] - z))
            worst = max(worst, err)
            count += 1
check("T1 fk o ik round trip", count > 100 and worst < 1e-6,
      f"{count} samples, worst {worst:.2e} m")

# ---------------------------------------------------------------- T2: pitch sign
# +pitch_cmd must rotate the foot targets nose-up in the BODY frame: front
# targets drop (front legs extend, front body rises), rear targets rise.
# Verified at the gait level with the real step() (the hip-frame route via
# body_local_orientation is a no-go: its z axis is remapped, pitch lands
# equally on all legs — measured -0.024/-0.026 front/rear).
gait = gc.TrotGaitController(default_stance, stance_time=0.04,
                             swing_time=0.18, time_step=0.02, use_imu=False)
s0 = gc.State(0.25)
s0.foot_locations = default_stance.copy()
s0.ticks = 40
c0 = gc.Command(0.25)
out0 = gait.step(s0, c0)
s1 = gc.State(0.25)
s1.foot_locations = default_stance.copy()
s1.ticks = 40
s1.pitch_cmd = 0.10
out1 = gait.step(s1, c0)
front_dz = out1[2, 0] - out0[2, 0]
rear_dz = out1[2, 2] - out0[2, 2]
check("T2 +pitch drops front targets / raises rear (nose-up)",
      front_dz < -0.005 and rear_dz > 0.005,
      f"front dz {front_dz:+.4f} rear dz {rear_dz:+.4f}")

# ---------------------------------------------------------------- T3: zero-bias equivalence
swing = gc.TrotSwingController(2, 9, 0.02, 11, 0.14, default_stance)
command = gc.Command(0.25)
state = gc.State(0.25)
state.foot_locations = default_stance.copy()
command.velocity = gc.np.array([0.03, 0.0, 0.0])
command.yaw_rate = gc.np.array([0.0, 0.0, 0.0])
ok = True
for prop in (0.1, 0.25, 0.5, 0.75, 0.9):   # subphase ticks stay inside (0, 1)
    state.swing_z_bias[:] = 0.0
    got = swing.next_foot_location(prop, 0, state, command)
    foot = state.foot_locations[:, 0]
    touchdown = swing.raibert_touchdown_location(0, command)
    time_left = swing.time_step * swing.swing_ticks * (1.0 - prop)
    legacy = (foot * gc.np.array([1, 1, 0])
              + gc.np.array([0, 0, swing.swing_height(prop) + command.robot_height])
              + (touchdown - foot) * gc.np.array([1, 1, 0]) / float(time_left)
              * swing.time_step)
    if not gc.np.allclose(got, legacy, atol=1e-15):
        ok = False
        print(f"  prop={prop} max diff {gc.np.abs(got - legacy).max():.2e}")
        break
check("T3 zero bias == legacy swing", ok)

# ---------------------------------------------------------------- helpers for T4-T6
def synth_joint_state(angles_per_leg):
    msg = gc.JointState()
    flat = []
    names = []
    for leg, a in enumerate(angles_per_leg):
        side = gc.SupportEstimator.LEG_SIDES[leg]
        names.extend([side + "_hip_joint", side + "_upper_leg_joint",
                      side + "_lower_leg_joint"])
        flat.extend(a)
    msg.name = names
    msg.position = flat
    return msg


def angles_for(foot_xy_z_per_leg):
    target = gc.np.zeros((3, 4))
    for leg, (x, y, z) in enumerate(foot_xy_z_per_leg):
        target[:, leg] = [x, y, z]
    a = ik.inverse_kinematics(target, 0, 0, 0, 0, 0, 0)
    return [tuple(a[leg * 3:leg * 3 + 3]) for leg in range(4)]


def new_estimator():
    return gc.SupportEstimator(ik, 0.25)


# ---------------------------------------------------------------- T4: flat no-op
est = new_estimator()
state = gc.State(0.25)
state.foot_locations = default_stance.copy()
state.ticks = 40  # a stance tick for all legs
command = gc.Command(0.25)
command.velocity = gc.np.array([0.04, 0.0, 0.0])
command.yaw_rate = gc.np.array([0.0, 0.0, 0.0])
nominal = [(x, y, -0.25) for (x, y) in NOMINAL_XY]
est.on_joint_states(synth_joint_state(angles_for(nominal)))
est.update(state, command, [1, 1, 1, 1])
flat_ok = (abs(est.ground_ref + 0.25) < 1e-6
           and abs(est.pitch_cmd) < 1e-9
           and not est.climbing
           and abs(est.swing_biases[0] - gc.STAIR_PROBE_BIAS) < 1e-9
           and est.swing_biases[2] == 0.0)
check("T4 flat: probe-only front bias, no pitch, not climbing", flat_ok,
      f"ground_ref {est.ground_ref:.3f} pitch {est.pitch_cmd:.4f}")

# ---------------------------------------------------------------- T5: stair response
est = new_estimator()
state = gc.State(0.25)
state.foot_locations = default_stance.copy()
command = gc.Command(0.25)
command.velocity = gc.np.array([0.04, 0.0, 0.0])
command.yaw_rate = gc.np.array([0.0, 0.0, 0.0])
# FL on a 15 cm step (-0.10), FR swinging blocked at ground level, rear nominal
angles = angles_for([
    (NOMINAL_XY[0][0], NOMINAL_XY[0][1], -0.25),   # FR: paw pressed on riser face
    (NOMINAL_XY[1][0], NOMINAL_XY[1][1], -0.10),   # FL: on the step
    (NOMINAL_XY[2][0], NOMINAL_XY[2][1], -0.25),
    (NOMINAL_XY[3][0], NOMINAL_XY[3][1], -0.25)])
est.on_joint_states(synth_joint_state(angles))
state.foot_locations[2, 0] = -0.09    # commanded swing target well above the paw
escalated_at = None
for tick in range(30):
    state.ticks = tick
    contact = [0, 1, 1, 1]            # FR swinging
    est.update(state, command, contact)
    if escalated_at is None and est.step_bias > 0.0:
        escalated_at = tick
climbing_response = (est.step_bias > 0.0 and est.pitch_cmd > 0.0
                     and est.climbing and est.swing_biases[2] > 0.0)
check("T5 stair: reflex escalates, nose-up, rear bias, climbing",
      climbing_response,
      f"bias {est.step_bias:.2f} (t={escalated_at}) pitch {est.pitch_cmd:.3f}")

# ---------------------------------------------------------------- T6: decay
before = est.step_bias
est.last_evidence = -1000
for tick in range(200, 500):
    state.ticks = tick
    est.update(state, command, [1, 1, 1, 1])   # all stance: no evidence
check("T6 bias decays without evidence", est.step_bias < before and est.step_bias <= 0.02,
      f"{before:.2f} -> {est.step_bias:.2f}")

# ---------------------------------------------------------------- T7: name mapping
# joint_states arrives in scrambled registration order (observed live on the
# rig); on_joint_states must map by joint NAME, not position.
est = new_estimator()
scrambled = ["lf_lower_leg_joint", "rf_hip_joint", "lf_hip_joint",
             "lh_upper_leg_joint", "lh_lower_leg_joint", "lf_upper_leg_joint",
             "rf_lower_leg_joint", "rh_hip_joint", "rf_upper_leg_joint",
             "rh_upper_leg_joint", "lh_hip_joint", "rh_lower_leg_joint"]
ref = angles_for([(x, y, -0.25 - 0.01 * leg) for leg, (x, y) in enumerate(NOMINAL_XY)])
msg = gc.JointState()
msg.name = scrambled
msg.position = [0.0] * 12
for leg, side in enumerate(("rf", "lf", "rh", "lh")):
    msg.position[scrambled.index(side + "_hip_joint")] = ref[leg][0]
    msg.position[scrambled.index(side + "_upper_leg_joint")] = ref[leg][1]
    msg.position[scrambled.index(side + "_lower_leg_joint")] = ref[leg][2]
est.on_joint_states(msg)
mapped_ok = all(est.angles[leg] is not None and
                max(abs(a - b) for a, b in zip(est.angles[leg], ref[leg])) < 1e-12
                for leg in range(4))
check("T7 scrambled joint_states mapped by name", mapped_ok)

print()
if FAILURES:
    print("FAILED:", ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
