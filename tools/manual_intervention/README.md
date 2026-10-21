# TinyNav manual intervention integration

This adapter feeds the Python TinyNav navigation stack through its Looper
bridge. Gazebo provides images, depth, and ground-truth camera poses. It is
not a VIO estimator and does not inject odometry faults.

The home setup uses the `yard` world, the `lekiwi` model, Gazebo Fortress,
`IGN_PARTITION=tinynav_home_42`, and ROS domain 42. Keep the ROS bridge, this
adapter, and TinyNav in the same container and DDS configuration.

## Model and bridge

`configure_world.py INPUT --output OUTPUT` replaces the expanded lekiwi
model's differential-drive plugin with Gazebo's VelocityControl system on
`/cmd_vel`. This is a planar kinematic control test: A/D really translate
sideways, rather than rotating a differential-drive model. It does not model
omnidirectional wheel dynamics. The input must already contain the expanded
lekiwi model and working camera/depth sensors. Do not modify `reference/`.

Bridge the following Gazebo topics into ROS:

| Gazebo topic | ROS topic | ROS type |
|---|---|---|
| `/sim/infra1` | `/sim/infra1` | sensor_msgs/msg/Image |
| `/sim/depth` | `/sim/depth` | sensor_msgs/msg/Image |
| `/camera/camera/color/image_raw` | same | sensor_msgs/msg/Image |
| `/clock` | `/clock` | rosgraph_msgs/msg/Clock |
| `/world/yard/dynamic_pose/info` | `/sim/world_pose` | tf2_msgs/msg/TFMessage |

Bridge ROS `/cmd_vel` (geometry_msgs/msg/Twist) to Gazebo `/cmd_vel`.
Run `python3 looper_source.py` in the ROS container. Its default output prefix
is `/camera/camera`, matching TinyNav's Looper bridge. Output uses consumer
clock stamps; camera/pose association uses the original simulation clock.

Start the TinyNav backend with:

```bash
export TINYNAV_SENSOR_MODE=looper TINYNAV_DISABLE_CHASSIS=1 ROBOT_TYPE=lekiwi
export TINYNAV_DB_PATH=/tinynav/tinynav_db/sim-home
python3 -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 3
```

## Verification

With the isolated simulator idle, run `python3 verify.py` inside the ROS
container using its DDS environment. It enters manual mode, confirms one
TinyNav command publisher, sends one command without heartbeat, and checks
that the backend times out and Gazebo stops. It exits manual mode on cleanup.

For task retention, enable Nav, wait for Localized, send POIs, then Take control.
Verify perception and planning keep updating, and manual arrival advances the
POI queue or completes the task. Exit manual must immediately hand an unfinished
task back to automatic control, without Resume. A task completed manually must
stay completed. Cancel explicitly ends the task.
