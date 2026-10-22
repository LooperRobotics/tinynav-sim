# Go2 simulation with the TinyNav application

Use the existing Go2 model with a fixed standing pose. Set
`TINYNAV_GO2_MOTION=kinematic` when running `gazebo/robots/go2/spawn.sh`.
In this mode the model is static and `/cmd_vel` updates its planar pose through
Gazebo's `set_pose` service. No gait, joint controllers, or velocity servo run.
Forward/backward, lateral motion, and yaw use body coordinates, with a 300 ms
command timeout. Movement follows simulation time and stops while paused.
This mode tests navigation and takeover, not leg dynamics or physical contact;
direct pose updates can pass through obstacles.

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
ros2_controllers for gait mode. Kinematic mode only needs the model,
robot_state_publisher, ros_gz_sim, and ros_gz_bridge. All ROS processes must share the same domain and DDS setup.
For host-network containers on one machine, the existing
`tools/probes/cyclone_localhost_unicast.xml` profile avoids cross-container
shared-memory transport issues.

Launch the existing Go2 world/spawn chain with the simulator's trajectory
follower and keyboard teleop disabled; TinyNav owns these commands. The
`gazebo/robots/go2/spawn.sh` starts the pose controller and service bridge
when `TINYNAV_GO2_MOTION=kinematic`; its default gait mode remains available.
Bridge the following Gazebo inputs into the application's ROS domain:

| Gazebo topic | ROS topic | ROS type |
|---|---|---|
| `/camera/camera/infra1/image_rect_raw` | `/sim/infra1` | sensor_msgs/msg/Image |
| `/camera/camera/infra1/camera_info` | `/sim/camera_info` | sensor_msgs/msg/CameraInfo |
| `/sim/depth` | `/sim/depth` | sensor_msgs/msg/Image |
| `/camera/camera/color/image_raw` | `/sim/color` | sensor_msgs/msg/Image |
| `/clock` | `/clock` | rosgraph_msgs/msg/Clock |
| `/world/<world>/pose/info` | `/sim/world_pose` | tf2_msgs/msg/TFMessage |

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

## Front and rear units

The Go2 xacro includes a second, identical 544x480 / 15 Hz stereo + RGB +
aligned-depth unit at the rear, with a 51 mm stereo baseline and 180-degree
mounting yaw. Both units include IMU sensors. Rear infra1 is at body
`(-0.19, -0.0255, 0.36)` metres; front infra1 is at `(0.19, 0.0255, 0.36)`.

Load the same ROS domain/DDS settings and `IGN_PARTITION` as Gazebo, then run:

```bash
bash tools/manual_intervention/bridge_looper.sh
python3 tools/manual_intervention/looper_source.py --rear
```

The bridge uses `TINYNAV_WORLD_NAME` (default `yard`). Native rear images are
remapped to `/sim/rear/{infra1,infra2,color,camera_info,depth}`; normalized Looper
outputs use `/camera/rear/`. A shared bounded timestamp cache makes frames from
the same simulation instant carry exactly the same output timestamp across
both units. Pose/image association still uses simulation time. Unique rear TF
aliases prevent the rear RGB transform overwriting the front mapping transform.
Without `--rear`, the adapter publishes the original front unit only.

In Pilot, enable the rear unit in Device > Cameras and save its installation.
Pilot records the rig manifest with the bag, stores both directions at each
keypose and fuses rear depth for occupancy/planning. The adapter still supplies
Gazebo ground truth, not independently estimated front/rear VIO.
