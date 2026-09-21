"""Sim-only site launch: gz world + robot + gz bridge + camera info + control.

The ros2-launch replacement of the sim half of sim/run_simulator.sh — one
lifecycle, launch-scoped env (no tmux bashrc reset traps), no navigation
stack. The stack starts separately and stays independent:

    ros2 launch sim/launch/sim.launch.py world:=sim/worlds/yard.sdf
    ros2 launch tinynav_cpp perception.launch.py        # this machine, if any
    # (a remote site would run orin_stack.launch.py behind the link)

The spawn-pose table and the per-robot gz-bridge topic list are ports of
run_simulator.sh; factory keeps its deck pose, other worlds spawn at origin.
"""
import os
import re
import subprocess

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS_ROOT = os.path.dirname(SIM_DIR)

#: Per-world go2 spawn poses (port of run_simulator.sh's case table): the
#: ros2_control initial_value holds the trot stance, so the pose must be
#: right at create time — a teleport after VIO init poisons the ISAM graph.
SPAWN_POSES = {
    ("factory", "lekiwi"): (-5.71, 2.68, 0.28, -1.57),
    ("factory", "go2"): (0.0, 0.0, 0.31, 0.0),
}


def world_name_of(world_sdf: str) -> str:
    with open(world_sdf) as f:
        match = re.search(r'<world name="([^"]+)"', f.read())
    if not match:
        raise RuntimeError(f"no <world name> in {world_sdf}")
    return match.group(1)


def dds_env_args(context) -> list:
    """Split-site DDS wiring: UDPv4-only builtin transports (the cross-
    container rehearsal has separate /dev/shm namespaces — SHM locators get
    announced but are unreachable, and the graph flaps; the real link is UDP
    anyway) and server-based unicast discovery (multicast is unreliable on
    wifi/point-to-point links, which is exactly what the real x86<->Orin
    link is). The discovery server process runs HERE, on the sim site."""
    return [
        SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
        SetEnvironmentVariable(
            "ROS_DISCOVERY_SERVER", LaunchConfiguration("discovery_server").perform(context)),
        ExecuteProcess(
            # The fastdds CLI is a shebang-less shell wrapper (bash runs it
            # via its ENOEXEC fallback; execvp does not) — call the python
            # entry point directly.
            cmd=["python3", "/opt/ros/humble/tools/fastdds/fastdds.py",
                 "discovery", "-i", "0"],
            name="discovery_server", output="screen"),
    ]


def gz_bridge_args(world_name: str, robot: str) -> list:
    """Port of msg_bridge_args: go2's /cmd_vel is consumed ROS-side by the
    gait chain (no gz diff-drive), so it is absent from the bridge there."""
    args = [
        "/camera/camera/infra1/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image",
        "/camera/camera/infra2/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image",
        "/camera/camera/color/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image",
        "/camera/camera/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU",
        "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        f"/world/{world_name}/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
    ]
    if robot != "go2":
        args.append("/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist")
    return args


def launch_setup(context, *args, **kwargs):
    world_sdf = LaunchConfiguration("world").perform(context)
    if not os.path.isabs(world_sdf):
        world_sdf = os.path.join(WS_ROOT, world_sdf)
    robot = LaunchConfiguration("robot").perform(context)
    world_name = world_name_of(world_sdf)
    x, y, z, yaw = SPAWN_POSES.get((world_name, robot), (0.0, 0.0, 0.0, 0.0))

    os.makedirs(os.path.join(WS_ROOT, "logs"), exist_ok=True)
    actions = [SetEnvironmentVariable("GDK_SCALE", "1")]
    actions += dds_env_args(context)
    actions += [
        SetEnvironmentVariable(
            "IGN_GAZEBO_RESOURCE_PATH",
            os.path.join(SIM_DIR, "scene", "models"),
        ),
        SetEnvironmentVariable("TINYNAV_WORLD_NAME", world_name),
        SetEnvironmentVariable("TINYNAV_ROBOT_MODEL", robot),
        SetEnvironmentVariable("ROBOT_TYPE", robot),
        # Procedural textures must exist before gz loads the world SDF.
        ExecuteProcess(cmd=
            ["python3", os.path.join(SIM_DIR, "scene", "gen_textures.py")],
            name="gen_textures", output="screen"),
        TimerAction(period=2.0, actions=[
            ExecuteProcess(cmd=
                ["ign", "gazebo", "-s", "-r", "--headless-rendering", "-v", "4",
                 world_sdf],
                name="gz_server", output="screen"),
            ExecuteProcess(cmd=
                ["ign", "gazebo", "-g", "-v", "3"],
                name="gz_gui", output="screen",
                condition=IfCondition(LaunchConfiguration("gui"))),
            ExecuteProcess(cmd=
                ["ros2", "run", "ros_gz_bridge", "parameter_bridge"]
                + gz_bridge_args(world_name, robot),
                name="gz_bridge", output="screen"),
            ExecuteProcess(cmd=
                ["python3", os.path.join(SIM_DIR, "scene", "camera_info_publisher.py")],
                name="camera_info", output="screen"),
        ]),
        # The robot window is the only spawn point; give the server time.
        TimerAction(period=6.0, actions=[
            ExecuteProcess(cmd=
                ["bash", os.path.join(SIM_DIR, "robots", robot, "spawn.sh")],
                additional_env={
                    "TINYNAV_SPAWN_X": str(x), "TINYNAV_SPAWN_Y": str(y),
                    "TINYNAV_SPAWN_Z": str(z), "TINYNAV_SPAWN_YAW": str(yaw),
                },
                name="robot_spawn", output="screen"),
            # Trajectory follower: turns /planning/trajectory_path into
            # /cmd_vel (go2: the servo consumes it; lekiwi: diff-drive).
            ExecuteProcess(cmd=
                ["python3",
                 os.path.join(WS_ROOT, "reference/tinynav/platforms/simulator_control.py")],
                name="simulator_control", output="screen",
                condition=IfCondition(LaunchConfiguration("control"))),
            ExecuteProcess(cmd=
                ["python3",
                 os.path.join(WS_ROOT, "reference/tinynav/platforms/keyboard_teleop.py")],
                name="teleop", output="screen",
                condition=IfCondition(LaunchConfiguration("teleop"))),
        ]),
    ]

    # Factory ships its plant model outside the repo (metaverse-source).
    factory_root = os.environ.get(
        "FACTORY_MODEL_ROOT", "/workspace/dm/metaverse-source/converted")
    if os.path.isdir(os.path.join(factory_root, "gz_model")):
        actions.insert(1, SetEnvironmentVariable(
            "IGN_GAZEBO_RESOURCE_PATH",
            factory_root + ":" + os.path.join(factory_root, "gz_model") + ":" +
            os.path.join(SIM_DIR, "scene", "models")))
    # go2 spawns from URDF with a ros2_control system plugin inside the
    # model; Fortress's SystemLoader only searches this path.
    if robot == "go2":
        actions.insert(1, SetEnvironmentVariable(
            "IGN_GAZEBO_SYSTEM_PLUGIN_PATH", "/opt/ros/humble/lib"))
    # gz rendering: PRIME offload when an NVIDIA GPU is present (port of the
    # run_simulator.sh probe).
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True)
        actions.insert(1, SetEnvironmentVariable("__GLX_VENDOR_LIBRARY_NAME", "nvidia"))
        actions.insert(1, SetEnvironmentVariable("__NV_PRIME_RENDER_OFFLOAD", "1"))
    except (OSError, subprocess.CalledProcessError):
        print("WARN: no NVIDIA GPU detected; sensor rendering may fail on old Mesa")
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "world", default_value=os.path.join(SIM_DIR, "worlds", "empty.sdf")),
        DeclareLaunchArgument("robot", default_value="go2"),
        DeclareLaunchArgument("control", default_value="true"),
        DeclareLaunchArgument("teleop", default_value="false"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("discovery_server", default_value="127.0.0.1:11811"),
        OpaqueFunction(function=launch_setup),
    ])
