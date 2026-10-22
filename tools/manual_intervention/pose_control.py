"""Drive a fixed-standing Gazebo model through planar pose updates."""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from ros_gz_interfaces.srv import SetEntityPose


class PoseControl(Node):
    def __init__(self, world):
        super().__init__('go2_pose_control')
        self.client = self.create_client(SetEntityPose, f'/world/{world}/set_pose')
        self.command = (0., 0., 0.)
        self.command_time = 0.
        self.clock = None
        self.previous_clock = None
        self.pose = None
        self.pending = None
        self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/sim/world_pose', self.on_pose, qos_profile_sensor_data)
        self.create_timer(.05, self.tick)

    def on_command(self, msg):
        command = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(v) for v in command):
            command = (0., 0., 0.)
        vx, vy, wz = command
        scale = min(1., .5 / max(math.hypot(vx, vy), 1e-9))
        self.command = (vx * scale, vy * scale, max(-1., min(1., wz)))
        self.command_time = time.monotonic()

    def on_clock(self, msg):
        now = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.clock is not None and now < self.clock:
            self.previous_clock = None
        self.clock = now

    def on_pose(self, msg):
        if self.pose is not None:
            return
        for transform in msg.transforms:
            if transform.child_frame_id == 'go2':
                t, q = transform.transform.translation, transform.transform.rotation
                self.pose = (t.x, t.y, 2 * math.atan2(q.z, q.w))
                break

    def tick(self):
        if self.clock is None or self.pose is None or not self.client.service_is_ready():
            return
        if self.pending is not None:
            return
        if self.previous_clock is None:
            self.previous_clock = self.clock
            return
        dt = max(0., min(.1, self.clock - self.previous_clock))
        self.previous_clock = self.clock
        vx, vy, wz = self.command if time.monotonic() - self.command_time < .3 else (0., 0., 0.)
        if dt == 0 or (vx == 0 and vy == 0 and wz == 0):
            return
        x, y, yaw = self.pose
        angle = yaw + wz * dt / 2
        target = (x + (vx * math.cos(angle) - vy * math.sin(angle)) * dt,
                  y + (vx * math.sin(angle) + vy * math.cos(angle)) * dt,
                  math.remainder(yaw + wz * dt, 2 * math.pi))
        req = SetEntityPose.Request()
        req.entity.name = 'go2'
        req.pose.position.x, req.pose.position.y = target[:2]
        req.pose.orientation.z = math.sin(target[2] / 2)
        req.pose.orientation.w = math.cos(target[2] / 2)
        self.pending = self.client.call_async(req)
        self.pending.add_done_callback(lambda future: self.on_result(future, target))

    def on_result(self, future, target):
        self.pending = None
        if future.exception() is not None:
            self.get_logger().error(f'Pose update failed: {future.exception()}')
        elif future.result().success:
            self.pose = target
        else:
            self.get_logger().error('Gazebo rejected the pose update')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = PoseControl(args.world)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
