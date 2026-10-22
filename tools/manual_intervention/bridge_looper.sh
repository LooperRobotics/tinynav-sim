#!/usr/bin/env bash
# Native Gazebo camera topics -> raw inputs for looper_source.py --rear.
set -e
source /opt/ros/humble/setup.bash
world=${TINYNAV_WORLD_NAME:-yard}
args=('/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'
      "/world/$world/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V")
remaps=(-r "/world/$world/pose/info:=/sim/world_pose")
for camera in camera rear; do
  raw=/sim
  if [[ $camera == rear ]]; then raw=/sim/rear; fi
  for stream in infra1 infra2 color; do
    suffix=image_rect_raw
    if [[ $stream == color ]]; then suffix=image_raw; fi
    topic=/camera/$camera/$stream/$suffix
    args+=("$topic@sensor_msgs/msg/Image[ignition.msgs.Image")
    remaps+=(-r "$topic:=$raw/$stream")
  done
  args+=("/camera/$camera/infra1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo")
  remaps+=(-r "/camera/$camera/infra1/camera_info:=$raw/camera_info")
  args+=("$raw/depth@sensor_msgs/msg/Image[ignition.msgs.Image")
done
args+=('/camera/camera/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU'
       '/camera/rear/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU')
exec ros2 run ros_gz_bridge parameter_bridge "${args[@]}" --ros-args "${remaps[@]}"
