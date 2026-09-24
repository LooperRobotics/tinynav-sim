"""gsplat sim-only site launch: 3DGS sensor server + ROS bridge + control.

The split-rig counterpart of gazebo/launch/sim.launch.py — same DDS wiring,
same remote_planning contract, one lifecycle, launch-scoped env (no tmux
bashrc reset traps), no navigation stack. The far site (Orin) runs
tinynav_cpp/launch/orin_stack.launch.py unchanged: the bridge publishes the
identical topic face as the gz bridge, so the link cannot tell the two
simulators apart.

    # x86 site (rig container):
    ros2 launch gsplat/launch/gsplat_sim.launch.py remote_planning:=true
    ros2 launch tinynav_cpp perception.launch.py     # this machine

    # Orin site (Humble, CycloneDDS pinned to the USB link):
    ros2 launch tinynav_cpp orin_stack.launch.py map_path:=<v2 map>

The sensor server MUST be launched with the gs venv + CUDA shim on its env
(gsplat checks for nvcc at import and refuses to load its compiled kernels
otherwise) — that env is set on the process here, so the launch also fixes
the tmux form's "re-export the world in every window" problem.
"""
import os

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

GS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS_ROOT = os.path.dirname(GS_DIR)

GS_VENV = os.environ.get("GS_VENV", "/opt/venv_gs")
GS_CUDA_HOME = os.environ.get("GS_CUDA_HOME", "/opt/cuda-shim-gs")
GS_PLAYGROUND_ROOT = os.environ.get(
    "GS_PLAYGROUND_ROOT", "/workspace/github/simulation/gs_playground")


def dds_env_args(context) -> list:
    """DDS wiring for the sim site — identical contract to sim.launch.py
    (cyclone default: XML pinned to the USB link NIC; fastdds legacy
    UDPv4 + discovery-server for single-distro rigs)."""
    if LaunchConfiguration("dds").perform(context) == "fastdds":
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
    return [
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        SetEnvironmentVariable(
            "CYCLONEDDS_URI",
            "file://" + os.path.join(
                WS_ROOT, "src", "tinynav_cpp", "config", "cyclonedds_x86.xml")),
    ]


def launch_setup(context, *args, **kwargs):
    scene = LaunchConfiguration("scene").perform(context)
    rtf = LaunchConfiguration("rtf").perform(context)

    os.makedirs(os.path.join(WS_ROOT, "logs"), exist_ok=True)
    # Trajectory follower command. remote_planning:=true (split rig): planning
    # runs on the Orin and its trajectory crosses the link as
    # /sim/trajectory_path; the follower is remapped onto that name here.
    control_cmd = [
        "python3",
        os.path.join(WS_ROOT, "reference/tinynav/platforms/simulator_control.py"),
    ]
    if LaunchConfiguration("remote_planning").perform(context) == "true":
        control_cmd += ["--ros-args", "-r",
                        "/planning/trajectory_path:=/sim/trajectory_path"]
    # The follower imports tinynav.core.robot_specs from the reference
    # snapshot (run_gsplat.sh exports this into every window; launch must
    # carry it on the process itself).
    control_pp = os.path.join(WS_ROOT, "reference")
    if "PYTHONPATH" in os.environ:
        control_pp += ":" + os.environ["PYTHONPATH"]

    # Sensor server: the 3DGS simulator itself (physics + RL policy + render).
    # CUDA env is required at import time — see the module docstring.
    server_cmd = [
        os.path.join(GS_VENV, "bin", "python"), "-u",
        os.path.join(GS_DIR, "server", "sensor_server.py"),
        "--config", scene,
        "--rtf", rtf,
    ]
    server_env = {
        "CUDA_HOME": GS_CUDA_HOME,
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "GS_PLAYGROUND_ROOT": GS_PLAYGROUND_ROOT,
    }
    gui_embed = LaunchConfiguration("gui_embed").perform(context) == "true"
    if gui_embed:
        # upstream-style viewer embedded in the sensor server (physics
        # viewport + camera/3DGS panels) — wgpu needs X11 + NVIDIA GL
        # `env -u`: an interactive parent shell sources /root/.bashrc, which
        # bakes the NVIDIA PRIME/EGL trio for rviz/gz — wgpu (the MotrixSim
        # viewer) probes NVIDIA EGL under it and PANICS; the viewer wants
        # plain X11 + Mesa (llvmpipe viewport).
        server_cmd = ["env", "-u", "__GLX_VENDOR_LIBRARY_NAME",
                      "-u", "__NV_PRIME_RENDER_OFFLOAD",
                      "-u", "__EGL_VENDOR_LIBRARY_FILENAMES"] + server_cmd
        # llvmpipe viewport ~24 ms/frame: cap the rate; panels off by
        # default (both are overridable with extra --server args)
        server_cmd += ["--window", "--window-fps", "15", "--window-panels", "off"]
        server_env["WAYLAND_DISPLAY"] = ""
        # llvmpipe complains but works without it; with it the per-frame
        # viewport sync measured ~3x cheaper (580 -> 197 ms/s)
        server_env.setdefault("XDG_RUNTIME_DIR", "/tmp")
    # Keep the shim's nvcc first on PATH (gsplat probes it); inherit the rest.
    server_env["PATH"] = os.path.join(GS_CUDA_HOME, "bin") + ":" + os.environ.get("PATH", "")

    actions = dds_env_args(context)
    actions += [
        ExecuteProcess(cmd=server_cmd, name="gs_sensor_server",
                       additional_env=server_env, output="screen"),
        # Viewer comes LAST and late: opening its window while the server is
        # still loading the scene reproducibly poisons the viewer's wgpu
        # present path (frame times jump to ~1 s and never recover, measured),
        # so give the server time to reach its steady loop first.
        TimerAction(period=25.0, actions=[
            # setsid + file redirect: same shape as run_gsplat.sh's tmux window,
            # which is the verified-good form (30 fps sustained).
            ExecuteProcess(cmd=["bash", "-c", " ".join([
                "setsid", "nohup", "env", "-i",
                f"DISPLAY={os.environ.get('DISPLAY', ':1')}",
                "XDG_RUNTIME_DIR=/tmp",
                f"HOME={os.environ.get('HOME', '/root')}",
                "PATH=/usr/bin:/bin",
                f"GS_PLAYGROUND_ROOT={GS_PLAYGROUND_ROOT}",
                os.path.join(GS_VENV, "bin", "python"), "-u",
                os.path.join(GS_DIR, "tools", "view_window.py"),
                "--scene", scene, "--fps", "30",
                "<", "/dev/null",
                ">", os.path.join(WS_ROOT, "logs", "gsview.log"), "2>&1", "&"]),
            ], name="gs_viewer", output="log",
                condition=IfCondition(LaunchConfiguration("gui"))),
        ]),
        # The bridge retries the ring internally; the delay just keeps the
        # "waiting for ring" warnings out of the log during scene load.
        TimerAction(period=8.0, actions=[
            ExecuteProcess(cmd=[
                "python3", "-u",
                os.path.join(GS_DIR, "ros", "gs_ros_bridge.py"),
                "--report-every", "15",
            ], name="gs_ros_bridge", output="screen"),
            ExecuteProcess(cmd=control_cmd,
                additional_env={"PYTHONPATH": control_pp},
                name="simulator_control", output="screen",
                condition=IfCondition(LaunchConfiguration("control"))),
            # Decoupled viewer window (gui:=true, the default path): its own
            # process reads the state slot, so the llvmpipe viewport never
            # steals the sim's main thread (measured rtf 0.92 + 25-30 fps view
            # vs 0.68 + 15 fps embedded).
            ExecuteProcess(cmd=
                ["python3",
                 os.path.join(WS_ROOT, "reference/tinynav/platforms/keyboard_teleop.py")],
                name="teleop", output="screen",
                condition=IfCondition(LaunchConfiguration("teleop"))),
        ]),
        ExecuteProcess(cmd=
            ["rviz2", "-d", os.path.join(WS_ROOT, "docs", "vis.rviz")],
            name="rviz", output="screen",
            condition=IfCondition(LaunchConfiguration("rviz"))),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scene", default_value="church"),
        DeclareLaunchArgument("rtf", default_value="1.0"),
        DeclareLaunchArgument("control", default_value="true"),
        DeclareLaunchArgument("teleop", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "gui", default_value="false",
            description="decoupled viewer window (own process, follows the state slot)"),
        DeclareLaunchArgument(
            "gui_embed", default_value="false",
            description="legacy in-process viewer inside the sensor server"),
        DeclareLaunchArgument("discovery_server", default_value="127.0.0.1:11811"),
        DeclareLaunchArgument("dds", default_value="cyclone"),
        DeclareLaunchArgument("remote_planning", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
