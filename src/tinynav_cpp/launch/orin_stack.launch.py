"""Orin site of the split deployment: the looper_bridge relay plus the
imu_propagator / mapping / planning components in one process.

    ros2 launch tinynav_cpp orin_stack.launch.py map_path:=/path/to/map_v2

The bridge subscribes the camera-box-namespace products once and republishes
the canonical /slam/* names locally (config/link_relay.yaml) — the mapping
component's ExactTime sync and planning's latest-depth both sit behind it.
use_sim_time:=true: every crossed stamp is sim time, so this site runs on
the sim clock carried across the link (clock crosses directly, like imu and
infra2 camera_info).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("tinynav_cpp")
    config = os.path.join(share, "config", "tinynav.yaml")
    relay = os.path.join(share, "config", "link_relay.yaml")
    # Split-site DDS wiring (the discovery server runs on the sim site):
    # UDPv4-only transports + server-based unicast discovery. The real link
    # (USB/Ethernet point-to-point) has no reliable multicast, and the
    # cross-container rehearsal has separate /dev/shm namespaces that make
    # the SHM builtin transport announce unreachable locators. On the real
    # link, pass discovery_server:=<sim site's link IP>:11811.
    actions = [
        DeclareLaunchArgument("params_file", default_value=config),
        DeclareLaunchArgument("map_path", default_value=""),
        DeclareLaunchArgument(
            "components", default_value="imu,mapping,planning"),
        DeclareLaunchArgument("discovery_server", default_value="127.0.0.1:11811"),
        SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
        SetEnvironmentVariable("ROS_DISCOVERY_SERVER",
                               LaunchConfiguration("discovery_server")),
        # Launch-scope env: the stack node reads it in main(); the bridge
        # ignores it.
        SetEnvironmentVariable("TINYNAV_COMPONENTS",
                               LaunchConfiguration("components")),
        Node(
            package="tinynav_cpp",
            executable="tinynav_bridge",
            arguments=[relay],
            output="screen",
        ),
        Node(
            package="tinynav_cpp",
            executable="tinynav_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file"),
                        {"map_path": LaunchConfiguration("map_path"),
                         "use_sim_time": True}],
        ),
    ]
    return LaunchDescription(actions)
