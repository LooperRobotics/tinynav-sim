# Launch the single-process C++ tinynav stack (all four components).
# Replaces: perception_node.py + map_node.py + planning_node.py + imu_propagator_node.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory('tinynav_cpp'), 'config', 'tinynav.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=config),
        # map format v2 directory (tools/export_map_v2.py output); empty = no map,
        # relocalization and global planning stay disabled.
        DeclareLaunchArgument('map_path', default_value=''),
        Node(
            package='tinynav_cpp',
            executable='tinynav_node',
            # No name= override: the four components keep their ported node
            # names (perception_node / map_node / planning_node /
            # imu_propagator_node) — same names as the Python stack, distinct
            # logger names for the per-node log files, and `ros2 param` can
            # address them individually again. tinynav.yaml matches via its
            # /** wildcard; intra-process comms match on topic+QoS, never on
            # node name (rclcpp IntraProcessManager, verified 2026-09-19).
            output='screen',
            parameters=[LaunchConfiguration('params_file'),
                        {'map_path': LaunchConfiguration('map_path')}],
        ),
    ])
