import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rclpy.init()
node = Node("probe_cap")
br = CvBridge()
got = {"kf": [], "infra": []}


def mk(key, n):
    def cb(msg):
        if len(got[key]) < n:
            img = br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            got[key].append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                             msg.encoding, msg.height, msg.width, img))
    return cb


node.create_subscription(Image, "/slam/keyframe_image", mk("kf", 2), 10)
node.create_subscription(Image, "/camera/camera/infra1/image_rect_raw", mk("infra", 2), 10)
end = node.get_clock().now().nanoseconds + 30_000_000_000
while node.get_clock().now().nanoseconds < end and (len(got["kf"]) < 2 or len(got["infra"]) < 2):
    rclpy.spin_once(node, timeout_sec=1)
for key in got:
    for i, (t, enc, h, w, img) in enumerate(got[key]):
        fn = f"/workspace/dm/tinynav-gazebo/fixtures/probe/{key}_{i}.png"
        cv2.imwrite(fn, img)
        print(key, i, f"t={t:.2f} enc={enc} {w}x{h} dtype={img.dtype} "
                      f"min={img.min()} max={img.max()} mean={img.mean():.1f} std={img.std():.1f}")
