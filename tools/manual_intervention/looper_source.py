"""Gazebo ground truth adapter for Looper bridge integration tests (not a VIO estimator)."""
import copy
import argparse
from tf2_ros import Buffer, TransformListener, TransformException
import cv2
from cv_bridge import CvBridge
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from tf2_msgs.msg import TFMessage
from rosgraph_msgs.msg import Clock
import message_filters


class SimLooper(Node):
    def __init__(self, prefix='/camera/camera', model='go2', camera_frame='camera_infra1_link'):
        super().__init__('sim_looper_source')
        self.model = model
        self.camera_frame = camera_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.extrinsic = None
        self.prefix = prefix.rstrip('/')
        self.bridge = CvBridge()
        self.camera_info = None
        self.create_subscription(CameraInfo, '/sim/camera_info', self.on_camera_info, qos_profile_sensor_data)
        self.color_pub = self.create_publisher(CompressedImage, self.prefix + '/color/image_rect_raw/compressed', 5)
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.on_color, qos_profile_sensor_data)
        self.poses = deque(maxlen=1000)
        self.clock = None
        self.clock_sub = self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.raw = self.create_subscription(TFMessage, '/sim/world_pose', self.on_pose, qos_profile_sensor_data)
        self.high_pub = self.create_publisher(PoseStamped, self.prefix + '/vio_100hz', 20)
        self.visual_pub = self.create_publisher(PoseStamped, self.prefix + '/vio_image', 10)
        self.image_pub = self.create_publisher(Image, self.prefix + '/infra1/image_rect_raw', 10)
        self.depth_pub = self.create_publisher(Image, self.prefix + '/depth/image_rect_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, self.prefix + '/infra1/camera_info', 10)
        self.left = message_filters.Subscriber(self, Image, '/sim/infra1', qos_profile=qos_profile_sensor_data)
        self.depth = message_filters.Subscriber(self, Image, '/sim/depth', qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.TimeSynchronizer([self.left,self.depth], 30)
        self.sync.registerCallback(self.on_images)
        self.create_timer(.01, self.on_timer)
        self.last_stamp = None
        self.frames = 0

    @staticmethod
    def seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_color(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            output = CompressedImage(); output.header = msg.header
            output.format = 'jpeg'; output.data = data.tobytes()
            self.color_pub.publish(output)

    def on_clock(self, msg):
        self.clock = msg.clock

    def on_pose(self, msg):
        if self.clock is None:
            return
        for tf in msg.transforms:
            if tf.child_frame_id == self.model:
                tf = copy.deepcopy(tf)
                tf.header.stamp = copy.deepcopy(self.clock)
                self.poses.append(tf)
                break

    def make_pose(self, tf, stamp):
        t,q = tf.transform.translation, tf.transform.rotation
        body = Rotation.from_quat([q.x,q.y,q.z,q.w])
        offset = self.extrinsic.transform.translation
        rotation = self.extrinsic.transform.rotation
        camera = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
        position = np.array([t.x,t.y,t.z]) + body.apply([offset.x,offset.y,offset.z])
        optical = body * camera * Rotation.from_matrix([[0,0,1],[-1,0,0],[0,-1,0]])
        pose = PoseStamped(); pose.header.stamp = stamp; pose.header.frame_id = 'world'
        pose.pose.position.x,pose.pose.position.y,pose.pose.position.z = map(float,position)
        pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w = map(float,optical.as_quat())
        return pose

    def on_timer(self):
        if self.extrinsic is None:
            try:
                self.extrinsic = self.tf_buffer.lookup_transform('base_link', self.camera_frame, rclpy.time.Time())
            except TransformException:
                return
        if not self.poses:
            return
        tf = self.poses[-1]
        stamp = self.seconds(self.clock)
        if stamp != self.last_stamp:
            self.high_pub.publish(self.make_pose(tf, self.get_clock().now().to_msg()))
            self.last_stamp = stamp

    def on_images(self, left, depth):
        if self.extrinsic is None or self.camera_info is None or not self.poses:
            return
        stamp = left.header.stamp
        tf = min(self.poses, key=lambda p:abs(self.seconds(p.header.stamp)-self.seconds(stamp)))
        if abs(self.seconds(tf.header.stamp)-self.seconds(stamp)) > .1:
            return
        # Match the real-time consumer clock even when simulation slows down.
        # Raw simulation stamps remain the basis for pose/image association.
        output_stamp = self.get_clock().now().to_msg()
        left = copy.deepcopy(left)
        depth = copy.deepcopy(depth)
        left.header.stamp = output_stamp
        depth.header.stamp = output_stamp
        info = copy.deepcopy(self.camera_info)
        info.header = copy.deepcopy(left.header)
        self.info_pub.publish(info)
        self.visual_pub.publish(self.make_pose(tf,output_stamp))
        self.depth_pub.publish(depth)
        self.image_pub.publish(left)
        self.frames+=1
        if self.frames % 150 == 1:
            self.get_logger().info(f'Published {self.frames} synchronized Looper frames; depth={depth.encoding}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='/camera/camera')
    parser.add_argument('--model', default='go2')
    parser.add_argument('--camera-frame', default='camera_infra1_link')
    args = parser.parse_args()
    rclpy.init(args=[]); node=SimLooper(args.output_prefix, args.model, args.camera_frame)
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': main()
