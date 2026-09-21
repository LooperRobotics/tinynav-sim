"""X86 site of the split deployment: the perception component alone, next to
the sim (start sim/launch/sim.launch.py first, same machine).

Perception's /slam/* products are remapped into the camera-box namespace so
they cross the link exactly once and collide with nothing on the Orin site —
whose looper_bridge relay owns the canonical names (fleet naming: the camera
publishes its products, the bridge renames them into /slam/*).

    ros2 launch tinynav_cpp perception.launch.py

/slam/camera_info and /slam/data have no Orin-side consumer and stay local;
imu, infra2 camera_info and /clock are consumed locally from the sim and
cross directly (single consumer each — see the relay table's header).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

#: Perception products that cross the link, in the camera-box namespace.
LINK_REMAPS = [
    ("/slam/odometry_visual", "/camera/camera/slam/odometry_visual"),
    ("/slam/keyframe_odom", "/camera/camera/slam/keyframe_odom"),
    ("/slam/keyframe_image", "/camera/camera/slam/keyframe_image"),
    ("/slam/keyframe_depth", "/camera/camera/slam/keyframe_depth"),
    ("/slam/depth", "/camera/camera/slam/depth"),
]


def generate_launch_description():
    share = get_package_share_directory("tinynav_cpp")
    config = os.path.join(share, "config", "tinynav.yaml")
    # Split-site DDS wiring (matches sim.launch.py, which runs the discovery
    # server): UDPv4-only transports + server-based unicast discovery. The
    # address points at the sim site — over the real USB/Ethernet link this
    # is the peer's link IP.
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=config),
        DeclareLaunchArgument("discovery_server", default_value="127.0.0.1:11811"),
        SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
        SetEnvironmentVariable("ROS_DISCOVERY_SERVER",
                               LaunchConfiguration("discovery_server")),
        Node(
            package="tinynav_cpp",
            executable="tinynav_node",
            output="screen",
            additional_env={"TINYNAV_COMPONENTS": "perception"},
            remappings=LINK_REMAPS,
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
