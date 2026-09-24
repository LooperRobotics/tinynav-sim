#!/usr/bin/env python3
"""Go2 trot controller for the tinynav sim rig: one file, no ROS packages.

Boots straight into a treadmill trot and stays in it; cmd_vel only adds
velocity on top. Safety patches carried here:
  - the cmd_vel clamp below: the open-loop plant has ~13x gain, the PI servo
    in cmd_vel_servo.py closes the loop, this is only a ceiling
  - the sqrt/D clamps in the IK: over-reach used to raise math domain
    errors and leave gz holding stale joint commands
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

RATE = 60  # Hz control loop (the trot math steps at time_step=0.02 s below)

# cmd_vel safety ceiling (see module docstring)
MAX_X_VELOCITY = 0.045
MAX_Y_VELOCITY = 0.015
MAX_YAW_RATE = 1.0


# ---------------------------------------------------------------------------
# rotation / homogeneous-transform helpers
# ---------------------------------------------------------------------------

def rotx(alpha):
    return np.array([[1, 0, 0],
                     [0, math.cos(alpha), -math.sin(alpha)],
                     [0, math.sin(alpha), math.cos(alpha)]])


def roty(beta):
    return np.array([[math.cos(beta), 0, math.sin(beta)],
                     [0, 1, 0],
                     [-math.sin(beta), 0, math.cos(beta)]])


def rotz(gamma):
    return np.array([[math.cos(gamma), -math.sin(gamma), 0],
                     [math.sin(gamma), math.cos(gamma), 0],
                     [0, 0, 1]])


def rotxyz(alpha, beta, gamma):
    return rotx(alpha).dot(roty(beta)).dot(rotz(gamma))


def homog_transxyz(dx, dy, dz):
    return np.array([[1, 0, 0, dx],
                     [0, 1, 0, dy],
                     [0, 0, 1, dz],
                     [0, 0, 0, 1]])


def homog_transform(dx, dy, dz, alpha, beta, gamma):
    rot4x4 = np.eye(4)
    rot4x4[:3, :3] = rotxyz(alpha, beta, gamma)
    return np.dot(homog_transxyz(dx, dy, dz), rot4x4)


def homog_transform_inverse(matrix):
    inverse = matrix
    inverse[:3, :3] = inverse[:3, :3].T
    inverse[:3, 3] = -np.dot(inverse[:3, :3], inverse[:3, 3])
    return inverse


# ---------------------------------------------------------------------------
# analytic leg IK (F/D clamps are load-bearing: see module docstring)
# ---------------------------------------------------------------------------

class InverseKinematics:
    def __init__(self, body_dimensions, leg_dimensions):
        self.bodyLength = body_dimensions[0]
        self.bodyWidth = body_dimensions[1]
        self.l1 = leg_dimensions[0]
        self.l2 = leg_dimensions[1]
        self.l3 = leg_dimensions[2]
        self.l4 = leg_dimensions[3]

    def get_local_positions(self, leg_positions, dx, dy, dz, roll, pitch, yaw):
        leg_positions = (np.block([[leg_positions], [np.array([1, 1, 1, 1])]])).T

        T_blwbl = homog_transform(dx, dy, dz, roll, pitch, yaw)

        T_blwFR1 = np.dot(T_blwbl, homog_transform(+0.5 * self.bodyLength,
                          -0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0))
        T_blwFL1 = np.dot(T_blwbl, homog_transform(+0.5 * self.bodyLength,
                          +0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0))
        T_blwRR1 = np.dot(T_blwbl, homog_transform(-0.5 * self.bodyLength,
                          -0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0))
        T_blwRL1 = np.dot(T_blwbl, homog_transform(-0.5 * self.bodyLength,
                          +0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0))

        pos_FR = np.dot(homog_transform_inverse(T_blwFR1), leg_positions[0])
        pos_FL = np.dot(homog_transform_inverse(T_blwFL1), leg_positions[1])
        pos_RR = np.dot(homog_transform_inverse(T_blwRR1), leg_positions[2])
        pos_RL = np.dot(homog_transform_inverse(T_blwRL1), leg_positions[3])

        return np.array([pos_FR[:3], pos_FL[:3], pos_RR[:3], pos_RL[:3]])

    def inverse_kinematics(self, leg_positions, dx, dy, dz, roll, pitch, yaw):
        positions = self.get_local_positions(leg_positions, dx, dy, dz, roll, pitch, yaw)
        angles = []

        for i in range(4):
            x = positions[i][0]
            y = positions[i][1]
            z = positions[i][2]

            F = math.sqrt(max(x**2 + y**2 - self.l2**2, 1e-9))
            G = F - self.l1
            H = math.sqrt(G**2 + z**2)

            theta1 = -math.atan2(y, x) - math.atan2(F, self.l2 * (-1)**i)

            D = (H**2 - self.l3**2 - self.l4**2) / (2 * self.l3 * self.l4)
            D = max(-0.999, min(0.999, D))

            theta4 = -math.atan2(math.sqrt(1 - D**2), D)

            theta3 = math.atan2(z, G) - math.atan2(self.l4 * math.sin(theta4),
                                                   self.l3 + self.l4 * math.cos(theta4))

            angles.append(theta1)
            angles.append(theta3)
            angles.append(theta4)

        # joint angles in radians for FR, FL, RR, RL
        return angles

    def foot_position_body(self, angles, leg_index):
        """Exact FK: (hip abduction, thigh, calf) angles -> foot (x, y, z) in
        the body frame (level body, world yaw irrelevant for support height).
        Algebraic inverse of inverse_kinematics() — fk_check.py asserts the
        round trip, do not edit one without the other.

        angles = (theta1, theta3, theta4) as produced by inverse_kinematics:
        theta1 in the xy plane, theta3/theta4 the knee-chain pair in the
        (F, z) plane with F = sqrt(x^2+y^2-l2^2).
        """
        t1, t3, t4 = angles
        l1, l2, l3, l4 = self.l1, self.l2, self.l3, self.l4

        # knee chain: from D = (H^2-l3^2-l4^2)/(2 l3 l4) and the IK branch
        # theta4 = -atan2(sqrt(1-D^2), D):  cos(theta4) = D
        h = math.sqrt(l3**2 + l4**2 + 2 * l3 * l4 * math.cos(t4))
        phi = t3 + math.atan2(l4 * math.sin(t4), l3 + l4 * math.cos(t4))
        g = h * math.cos(phi)
        z = h * math.sin(phi)
        f = g + l1

        # planar pair: theta1 = -atan2(y, x) - atan2(F, l2*(-1)^leg_index),
        # and F = sqrt(x^2+y^2-l2^2)  =>  |(x, y)| = sqrt(F^2 + l2^2)
        s = l2 * (-1) ** leg_index
        psi = -t1 - math.atan2(f, s)
        r_xy = math.sqrt(f**2 + l2**2)
        x = r_xy * math.cos(psi)
        y = r_xy * math.sin(psi)

        # hip-local frame -> body frame (same per-leg transform the IK
        # inverts; body pose is identity — support estimation only needs the
        # physical body frame, not the commanded one)
        if leg_index == 0:
            t = homog_transform(+0.5 * self.bodyLength, -0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0)
        elif leg_index == 1:
            t = homog_transform(+0.5 * self.bodyLength, +0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0)
        elif leg_index == 2:
            t = homog_transform(-0.5 * self.bodyLength, -0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0)
        else:
            t = homog_transform(-0.5 * self.bodyLength, +0.5 * self.bodyWidth, 0, math.pi / 2, -math.pi / 2, 0)
        p = np.dot(t, np.array([x, y, z, 1.0]))
        return p[:3]


# ---------------------------------------------------------------------------
# stair adaptation (Route A, proprioceptive "blind" climbing)
# ---------------------------------------------------------------------------

# knobs (kept plain so fk_check.py can exercise the same numbers)
STAIR_PROBE_BIAS = 0.02     # constant front-paw probe raise while driving (m)
STAIR_BIAS_MAX = 0.18       # reflex escalation ceiling (covers 15 cm risers + clearance)
STAIR_BIAS_STEP = 0.02      # per-escalation raise (m)
STAIR_BLOCK_TICKS = 5       # blocked ticks on one leg before escalating
STAIR_BLOCK_MARGIN = 0.02   # commanded-above-actual z that counts as blocked (m)
STAIR_DECAY_TICKS = 150     # ~3 s without evidence before the bias decays
STAIR_PITCH_K = 1.2         # rad per m of front/rear support difference
STAIR_PITCH_MAX = 0.17      # 10 deg clamp
STAIR_PITCH_RATE = 0.0018   # rad per 20 ms tick (~5 deg/s)
STAIR_PITCH_DEADBAND = 0.015
STAIR_CLIMB_ENTER = 0.02    # front/rear diff (m) that flags climbing
STAIR_CLIMB_EXIT_TICKS = 100
STAIR_SPEED_GATE = 0.02     # vx ceiling while climbing (m/s)
STAIR_SWING_LIFT = 0.14     # the trot's z_leg_lift (sets the apex headroom)
STAIR_APEX_MAX = 0.05       # swing apex body-frame z ceiling (leg-fold limit)


class SupportEstimator:
    """Support-surface tracking from actual joint angles, plus the blocked-paw
    reflex that discovers each riser.

    All outputs are additive corrections on top of the legacy gait:
      swing_z_bias  per-leg raise of the swing touchdown height — on flat it
                    is the small front probe (STAIR_PROBE_BIAS), on stairs it
                    grows to the measured/estimated riser;
      pitch_cmd     commanded body pitch (nose-up on ascent) fed through
                    state.body_local_orientation;
      climbing      gates the vx ceiling.

    With no joint_states yet, or on flat ground, everything degrades to the
    legacy behavior (bias = probe only for front legs, pitch = 0).
    """

    def __init__(self, ik, default_height):
        self.ik = ik
        self.h_nom = default_height                # positive magnitude
        self.angles = [None] * 4                   # (t1, t3, t4) per leg
        self.foot_z = [-default_height] * 4        # body-frame foot z (negative)
        self.ground_ref = -default_height          # median stance support z
        self.front_support = -default_height
        self.rear_support = -default_height
        self.swing_biases = np.zeros(4)
        self.step_bias = 0.0                       # reflex-escalated rise estimate
        self.pitch_cmd = 0.0
        self.block_counters = [0, 0, 0, 0]
        self.last_evidence = -10 ** 9
        self.climb_counter = 0
        self.climbing = False

    # gait leg order FR, FL, RR, RL -> urdf side prefix; joint_states arrives
    # in a scrambled registration order, so map by NAME, never by index
    LEG_SIDES = ("rf", "lf", "rh", "lh")

    def on_joint_states(self, msg):
        # actual encoder positions (gz physics feedback), not commanded ones
        if len(msg.position) < 12 or len(msg.name) < 12:
            return
        idx = {name: i for i, name in enumerate(msg.name)}
        for leg, side in enumerate(self.LEG_SIDES):
            try:
                self.angles[leg] = (
                    msg.position[idx[side + "_hip_joint"]],
                    msg.position[idx[side + "_upper_leg_joint"]],
                    msg.position[idx[side + "_lower_leg_joint"]])
            except KeyError:
                return

    def update(self, state, command, contact_modes):
        ticks = state.ticks
        commanded_z = state.foot_locations[2]

        # FK the actual leg configuration (nominal fallback until the first
        # joint_states arrives — corrections stay at legacy values)
        for leg in range(4):
            if self.angles[leg] is not None:
                self.foot_z[leg] = self.ik.foot_position_body(self.angles[leg], leg)[2]

        stance = [leg for leg in range(4) if contact_modes[leg] == 1]
        stance = stance if stance else list(range(4))
        self.ground_ref = float(np.median([self.foot_z[leg] for leg in stance]))
        front = [self.foot_z[leg] for leg in (0, 1) if contact_modes[leg] == 1]
        rear = [self.foot_z[leg] for leg in (2, 3) if contact_modes[leg] == 1]
        self.front_support = float(np.median(front)) if front else self.ground_ref
        self.rear_support = float(np.median(rear)) if rear else self.ground_ref

        # blocked-paw reflex: a swinging paw whose commanded target sits
        # STAIR_BLOCK_MARGIN above the FK position for STAIR_BLOCK_TICKS ticks
        # is pressing a riser face — raise the shared rise estimate
        escalated = False
        for leg in range(4):
            if contact_modes[leg] == 0 and self.angles[leg] is not None:
                blocked = commanded_z[leg] - self.foot_z[leg]
                if blocked > STAIR_BLOCK_MARGIN:
                    self.block_counters[leg] += 1
                    if self.block_counters[leg] >= STAIR_BLOCK_TICKS:
                        self.step_bias = min(self.step_bias + STAIR_BIAS_STEP, STAIR_BIAS_MAX)
                        self.block_counters = [0, 0, 0, 0]
                        self.last_evidence = ticks
                        escalated = True
                        break
                else:
                    self.block_counters[leg] = 0
        if not escalated and ticks - self.last_evidence > STAIR_DECAY_TICKS \
                and self.step_bias > 0.0 and ticks % 10 == 0:
            self.step_bias = max(0.0, self.step_bias - 0.01)

        # per-leg swing bias: probe on the front pair while driving (flat-safe,
        # escalates via the reflex); once a rise is discovered every leg gets
        # it — rear paws must also land on the raised tread. Capped so the
        # swing apex (ground_ref + lift + bias) never crosses the body plane
        # and saturates the IK.
        forward = abs(command.velocity[0]) > 0.005 or abs(command.velocity[1]) > 0.005
        base_bias = self.step_bias if self.step_bias > 0.0 else (
            STAIR_PROBE_BIAS if forward else 0.0)
        # apex = robot_height + lift + bias must stay under the fold limit
        apex_cap = max(0.0, STAIR_APEX_MAX - (-self.h_nom + STAIR_SWING_LIFT))
        applied = min(base_bias, apex_cap)
        for leg in range(4):
            self.swing_biases[leg] = applied if (leg in (0, 1) or self.step_bias > 0.0) else 0.0

        # pitch: nose into the ascent, rate-limited, deadbanded, clamped
        diff = self.front_support - self.rear_support
        if abs(diff) > STAIR_PITCH_DEADBAND:
            target = max(-STAIR_PITCH_MAX, min(STAIR_PITCH_MAX, STAIR_PITCH_K * diff))
        else:
            target = 0.0
        step = max(-STAIR_PITCH_RATE, min(STAIR_PITCH_RATE, target - self.pitch_cmd))
        self.pitch_cmd += step

        # climbing flag with exit hysteresis (gates the vx ceiling)
        climbing_now = self.climbing or self.step_bias > 0.02 or diff > STAIR_CLIMB_ENTER
        if climbing_now:
            self.climb_counter = STAIR_CLIMB_EXIT_TICKS
            self.climbing = True
        else:
            self.climb_counter -= 1
            if self.climb_counter <= 0:
                self.climbing = False

    def speed_gate(self, vx):
        return min(vx, STAIR_SPEED_GATE) if self.climbing else vx


class State:
    def __init__(self, default_height):
        self.velocity = np.array([0.0, 0.0])
        self.yaw_rate = 0.0
        self.robot_height = -default_height
        self.foot_locations = np.zeros((3, 4))
        self.body_local_position = np.array([0.0, 0.0, 0.0])
        self.body_local_orientation = np.array([0.0, 0.0, 0.0])
        self.imu_roll = 0.0
        self.imu_pitch = 0.0
        self.ticks = 0
        # stair adaptation (Route A): per-leg additive swing-z raise plus a
        # commanded body pitch. Kept at zero unless SupportEstimator writes
        # them — zeros reproduce the legacy gait bit-for-bit.
        self.swing_z_bias = np.zeros(4)
        self.pitch_cmd = 0.0


class Command:
    def __init__(self, robot_height):
        self.robot_height = -robot_height
        self.velocity = np.array([0.0, 0.0, 0.0])   # [x, y, z]
        self.yaw_rate = np.array([0.0, 0.0, 0.0])   # [roll, pitch, yaw]


class PID_controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.desired_roll_pitch = np.array([0.0, 0.0])
        self.I_term = np.array([0.0, 0.0])
        self.D_term = np.array([0.0, 0.0])
        self.max_I = 0.2
        self.last_error = np.array([0.0, 0.0])
        self.last_time = None

    def run(self, roll, pitch):
        error = self.desired_roll_pitch - np.array([roll, pitch])

        t_now = Time().seconds_nanoseconds()[0] + Time().seconds_nanoseconds()[1] * 1e-9
        if self.last_time is None:
            self.last_time = t_now
            return np.array([0.0, 0.0])

        step = t_now - self.last_time
        if step < 1e-6:
            return np.array([0.0, 0.0])

        self.I_term += error * step
        for i in range(2):
            if self.I_term[i] < -self.max_I:
                self.I_term[i] = -self.max_I
            elif self.I_term[i] > self.max_I:
                self.I_term[i] = self.max_I

        self.D_term = (error - self.last_error) / step

        self.last_time = t_now
        self.last_error = error

        return self.kp * error + self.I_term * self.ki + self.D_term * self.kd

    def reset(self):
        self.last_time = Time().seconds_nanoseconds()[0] + Time().seconds_nanoseconds()[1] * 1e-9
        self.I_term = np.array([0.0, 0.0])
        self.D_term = np.array([0.0, 0.0])
        self.last_error = np.array([0.0, 0.0])


# ---------------------------------------------------------------------------
# 60 Hz trot: phase sequencer + swing/stance foot controllers
# ---------------------------------------------------------------------------

class GaitController:
    def __init__(self, stance_time, swing_time, time_step, contact_phases, default_stance):
        self.stance_time = stance_time
        self.swing_time = swing_time
        self.time_step = time_step
        self.contact_phases = contact_phases
        self.def_stance = default_stance

    @property
    def default_stance(self):
        return self.def_stance

    @property
    def stance_ticks(self):
        return int(self.stance_time / self.time_step)

    @property
    def swing_ticks(self):
        return int(self.swing_time / self.time_step)

    @property
    def phase_ticks(self):
        temp = []
        for i in range(len(self.contact_phases[0])):
            if 0 in self.contact_phases[:, i]:
                temp.append(self.swing_ticks)
            else:
                temp.append(self.stance_ticks)
        return temp

    @property
    def phase_length(self):
        return sum(self.phase_ticks)

    def phase_index(self, ticks):
        phase_time = ticks % self.phase_length
        phase_sum = 0
        phase_ticks = self.phase_ticks
        for i in range(len(self.contact_phases[0])):
            phase_sum += phase_ticks[i]
            if phase_time < phase_sum:
                return i
        assert False

    def subphase_ticks(self, ticks):
        phase_time = ticks % self.phase_length
        phase_sum = 0
        phase_ticks = self.phase_ticks
        for i in range(len(self.contact_phases[0])):
            phase_sum += phase_ticks[i]
            if phase_time < phase_sum:
                return phase_time - phase_sum + phase_ticks[i]
        assert False

    def contacts(self, ticks):
        return self.contact_phases[:, self.phase_index(ticks)]


class TrotSwingController:
    def __init__(self, stance_ticks, swing_ticks, time_step, phase_length, z_leg_lift, default_stance):
        self.stance_ticks = stance_ticks
        self.swing_ticks = swing_ticks
        self.time_step = time_step
        self.phase_length = phase_length
        self.z_leg_lift = z_leg_lift
        self.default_stance = default_stance

    def raibert_touchdown_location(self, leg_index, command):
        delta_pos_2d = command.velocity * self.phase_length * self.time_step * 1.0
        delta_pos = np.array([delta_pos_2d[0], delta_pos_2d[1], 0])

        theta = self.stance_ticks * self.time_step * command.yaw_rate[2]
        rotation = rotz(theta)

        return np.matmul(rotation, self.default_stance[:, leg_index]) + delta_pos

    def swing_height(self, swing_phase):
        if swing_phase < 0.5:
            return (swing_phase / 0.5) * self.z_leg_lift * 1.0
        return self.z_leg_lift * (1 - (swing_phase - 0.5) / 0.5) * 1.0

    def next_foot_location(self, swing_prop, leg_index, state, command):
        assert 0 <= swing_prop <= 1
        foot_location = state.foot_locations[:, leg_index]
        swing_height_ = self.swing_height(swing_prop)
        touchdown_location = self.raibert_touchdown_location(leg_index, command)

        time_left = self.time_step * self.swing_ticks * (1.0 - swing_prop)
        velocity = (touchdown_location - foot_location) / float(time_left) * np.array([1, 1, 0])

        delta_foot_location = velocity * self.time_step
        # stair adaptation rides in as an additive per-leg bias; the zero
        # vector reproduces the legacy trajectory exactly (fk_check T3)
        z_vector = np.array([0, 0, swing_height_ + command.robot_height
                             + state.swing_z_bias[leg_index]])
        return foot_location * np.array([1, 1, 0]) + z_vector + delta_foot_location


class TrotStanceController:
    def __init__(self, phase_length, stance_ticks, swing_ticks, time_step, z_error_constant):
        self.phase_length = phase_length
        self.stance_ticks = stance_ticks
        self.swing_ticks = swing_ticks
        self.time_step = time_step
        self.z_error_constant = z_error_constant

    def position_delta(self, leg_index, state, command):
        z = state.foot_locations[2, leg_index]

        step_dist_x = command.velocity[0] * (float(self.phase_length) / self.swing_ticks)
        step_dist_y = command.velocity[1] * (float(self.phase_length) / self.swing_ticks)

        velocity = np.array([
            -(step_dist_x / 4) / (float(self.time_step) * self.stance_ticks),
            -(step_dist_y / 4) / (float(self.time_step) * self.stance_ticks),
            1.0 / self.z_error_constant * (state.robot_height - z)
        ])

        delta_pos = velocity * self.time_step
        delta_ori = rotxyz(
            -command.yaw_rate[0] * self.time_step,
            -command.yaw_rate[1] * self.time_step,
            -command.yaw_rate[2] * self.time_step
        )
        return (delta_pos, delta_ori)

    def next_foot_location(self, leg_index, state, command):
        foot_location = state.foot_locations[:, leg_index]
        (delta_pos, delta_ori) = self.position_delta(leg_index, state, command)
        return np.matmul(delta_ori, foot_location) + delta_pos


class TrotGaitController(GaitController):
    def __init__(self, default_stance, stance_time, swing_time, time_step, use_imu):
        self.use_imu = use_imu
        self.autoRest = True
        self.trotNeeded = True

        contact_phases = np.array([[1, 1, 1, 0],  # 0: Leg swing
                                   [1, 0, 1, 1],  # 1: Moving stance forward
                                   [1, 0, 1, 1],
                                   [1, 1, 1, 0]])

        z_error_constant = 0.02
        z_leg_lift = 0.14

        super().__init__(stance_time, swing_time, time_step, contact_phases, default_stance)

        self.swingController = TrotSwingController(
            self.stance_ticks, self.swing_ticks, self.time_step,
            self.phase_length, z_leg_lift, self.default_stance)

        self.stanceController = TrotStanceController(
            self.phase_length, self.stance_ticks, self.swing_ticks,
            self.time_step, z_error_constant)

        #                                       kp    ki    kd
        self.pid_controller = PID_controller(0.15, 0.02, 0.002)

    def step(self, state, command):
        if self.autoRest:
            if command.velocity[0] == 0 and command.velocity[1] == 0 and np.all(command.yaw_rate == 0):
                if state.ticks % (2 * self.phase_length) == 0:
                    self.trotNeeded = False
            else:
                self.trotNeeded = True

        if self.trotNeeded:
            contact_modes = self.contacts(state.ticks)

            new_foot_locations = np.zeros((3, 4))
            for leg_index in range(4):
                contact_mode = contact_modes[leg_index]
                if contact_mode == 1:
                    new_location = self.stanceController.next_foot_location(leg_index, state, command)
                else:
                    swing_proportion = float(self.subphase_ticks(state.ticks)) / float(self.swing_ticks)
                    new_location = self.swingController.next_foot_location(swing_proportion, leg_index, state, command)

                new_foot_locations[:, leg_index] = new_location

            if self.use_imu:
                compensation = self.pid_controller.run(state.imu_roll, state.imu_pitch)
                rot = rotxyz(-compensation[0], -compensation[1], 0)
                new_foot_locations = np.matmul(rot, new_foot_locations)

            # stair adaptation: commanded nose-up pitch — rotate the foot
            # targets in the body frame so front targets drop (front legs
            # extend) and rear targets rise (rear legs fold). Zero pitch skips
            # the matmul entirely (bit-exact legacy).
            if state.pitch_cmd != 0.0:
                new_foot_locations = np.matmul(roty(state.pitch_cmd),
                                               new_foot_locations)

            state.ticks += 1
            return new_foot_locations
        else:
            temp = self.default_stance.copy()
            temp[2] = [command.robot_height] * 4
            return temp

    def run(self, state, command):
        state.foot_locations = self.step(state, command)
        state.robot_height = command.robot_height
        return state.foot_locations


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------

class Go2Controller(Node):
    def __init__(self):
        super().__init__("go2_controller")

        body = [0.3762, 0.0935]
        legs = [0.0, 0.0955, 0.213, 0.213]

        self.inverseKinematics = InverseKinematics(body, legs)

        delta_x = body[0] * 0.5
        delta_y = body[1] * 0.5 + legs[1]
        # FR, FL, RR, RL
        default_stance = np.array([
            [delta_x + 0.02, delta_x + 0.02, -delta_x + -0.0, -delta_x + -0.0],
            [-delta_y, delta_y, -delta_y, delta_y],
            [0, 0, 0, 0]
        ])

        self.state = State(0.25)
        self.state.foot_locations = default_stance
        self.command = Command(0.25)

        # boots straight into a treadmill trot; trots in place until
        # cmd_vel arrives
        self.trot = TrotGaitController(default_stance, stance_time=0.04,
                                       swing_time=0.18, time_step=0.02, use_imu=True)
        self.trot.pid_controller.reset()

        # stair adaptation (Route A): proprioceptive support tracking. Off by
        # default until the flat-regression pass signs off; False = the
        # estimator is never updated and every correction stays zero. Live
        # tunable via ros2 param set (no restart).
        self.declare_parameter("stair_adapt", False)
        self.stair_adapt = self.get_parameter("stair_adapt").value
        self.add_on_set_parameters_callback(self._on_param)
        self.estimator = SupportEstimator(self.inverseKinematics, 0.25)
        self.stair_debug_publisher = self.create_publisher(
            Float64MultiArray, "stair_debug", 10)
        self.debug_tick = 0

        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray, "joint_group_controller/commands", 10)

        # /cmd_vel is the PI servo's output (cmd_vel_servo.py); desired
        # velocity arrives as /cmd_vel (absolute) and reaches this node only
        # through the PI servo's namespaced output
        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(JointState, "joint_states", self.estimator.on_joint_states, 10)

        self.timer = self.create_timer(1.0 / RATE, self.control_loop)

    def _on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "stair_adapt":
                self.stair_adapt = p.value
                if not p.value:
                    self.state.swing_z_bias[:] = 0.0
                    self.state.pitch_cmd = 0.0
        return SetParametersResult(successful=True)

    def on_cmd_vel(self, msg):
        self.command.velocity = np.array([
            min(max(msg.linear.x, -MAX_X_VELOCITY), MAX_X_VELOCITY),
            min(max(msg.linear.y, -MAX_Y_VELOCITY), MAX_Y_VELOCITY),
            msg.linear.z
        ])
        self.command.yaw_rate = np.array([
            msg.angular.x,
            msg.angular.y,
            min(max(msg.angular.z, -MAX_YAW_RATE), MAX_YAW_RATE)
        ])

    def control_loop(self):
        # stair adaptation: update corrections from the last tick's commands
        # and the actual joint angles, then hand them to the gait through the
        # additive state fields. stair_adapt=False leaves every field at its
        # legacy value (zeros).
        if self.stair_adapt:
            contact_modes = self.trot.contacts(self.state.ticks)
            self.estimator.update(self.state, self.command, contact_modes)
            self.state.swing_z_bias[:] = self.estimator.swing_biases
            self.state.pitch_cmd = self.estimator.pitch_cmd
            self.command.velocity[0] = self.estimator.speed_gate(self.command.velocity[0])

        leg_positions = self.trot.run(self.state, self.command)

        # observability for the gait verification runs (6 Hz)
        self.debug_tick += 1
        if self.debug_tick % 10 == 0:
            e = self.estimator
            dbg = Float64MultiArray()
            dbg.data = [e.step_bias, e.pitch_cmd, float(e.climbing),
                        e.front_support, e.rear_support, e.ground_ref]
            self.stair_debug_publisher.publish(dbg)

        dx, dy, dz = self.state.body_local_position
        roll, pitch, yaw = self.state.body_local_orientation

        try:
            joint_angles = self.inverseKinematics.inverse_kinematics(
                leg_positions, dx, dy, dz, roll, pitch, yaw)
            msg = Float64MultiArray()
            msg.data = joint_angles
            self.joint_command_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Go2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
