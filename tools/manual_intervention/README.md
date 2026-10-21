# Go2 simulation with the TinyNav application

Use the existing Go2 model, joint controllers, gait controller, and velocity
servo under `gazebo/robots/go2/`. Automatic and manual TinyNav commands both
enter `/cmd_vel`; the existing servo feeds `/robot1/cmd_vel`. Do not bridge
`/cmd_vel` to a separate Gazebo motion plugin or run another trajectory follower.
The servo handles forward velocity, lateral velocity, and yaw through the same
Go2 gait. Its existing skating-gait limitations still apply.

The Go2 sensor head also publishes aligned depth on `/sim/depth` for the
Looper input adapter. `looper_source.py` converts images, depth, and Gazebo
ground-truth poses to Looper-compatible topics. It uses the robot's TF camera
extrinsics and Gazebo CameraInfo, rather than model-specific camera offsets.
Color, infrared, depth, and visual poses share one output timestamp after
association on the simulation clock. The adapter also publishes color camera
intrinsics and the Looper camera transform used by offline mapping.
This adapter supplies ground truth; it does not estimate VIO or inject faults.

## Start

Run inside the development/simulation containers with the ROS environment
loaded. The Gazebo container needs the existing Go2 dependencies: xacro,
robot_state_publisher, ros_gz_sim, ign_ros2_control, controller_manager, and
ros2_controllers. All ROS processes must share the same domain and DDS setup.
For host-network containers on one machine, the existing
`tools/probes/cyclone_localhost_unicast.xml` profile avoids cross-container
shared-memory transport issues.

Launch the existing Go2 world/spawn chain with the simulator's trajectory
follower and keyboard teleop disabled; TinyNav owns these commands. The
existing `gazebo/robots/go2/spawn.sh` starts the Go2 gait and velocity servo.
Bridge the following Gazebo inputs into the application's ROS domain:

| Gazebo topic | ROS topic | ROS type |
|---|---|---|
| `/camera/camera/infra1/image_rect_raw` | `/sim/infra1` | sensor_msgs/msg/Image |
| `/camera/camera/infra1/camera_info` | `/sim/camera_info` | sensor_msgs/msg/CameraInfo |
| `/sim/depth` | `/sim/depth` | sensor_msgs/msg/Image |
| `/camera/camera/color/image_raw` | `/sim/color` | sensor_msgs/msg/Image |
| `/clock` | `/clock` | rosgraph_msgs/msg/Clock |
| `/world/<world>/dynamic_pose/info` | `/sim/world_pose` | tf2_msgs/msg/TFMessage |

Run `python3 looper_source.py`. Defaults are model `go2`, body frame
`base_link`, camera frame `camera_infra1_link`, and output prefix
`/camera/camera`. Model and camera frame can be selected with `--model` and
`--camera-frame`. The robot_state_publisher supplies the camera TF.

Start the application backend with the stock Go2 configuration:

```bash
export ROBOT_TYPE=go2
export TINYNAV_DB_PATH=/tinynav/tinynav_db/sim-home-go2
python3 run_tinynav.py
```

The launcher only selects Looper input and skips the physical chassis driver;
it does not replace robot geometry or node launch commands in TinyNav.
Build the web frontend using the normal TinyNav workflow. Use a map recorded
with the Go2 sensor head; maps made with a different camera height should be
kept separately.

Take control keeps perception, planning, and POI tracking running. Manual
arrival can advance or complete a task. Exit manual hands an unfinished task
back to automatic control without Continue.
