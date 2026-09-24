# tinynav-sim

Independent workspace for the TinyNav C++ rewrite: gzsim simulation + Python
reference snapshot + the `tinynav_cpp` ROS2 package (single process, multiple
components, intra-process zero-copy communication).

## Layout

```
gazebo/                gzsim worlds / robots / scripted scenes + run_simulator.sh
                    (extracted from tinynav/tool/simulator; stack-agnostic)
reference/tinynav/  Python reference snapshot (core/*.py + cpp/*.cpp) of branch
                    gzy/fix/gzsim — the spec the C++ port must match. Read-only.
src/tinynav_cpp/    the C++ package:
  src/core/         pure C++ math (SE3, PnP, IMU integration)
  src/kernels/      raycast / pose-graph / BA ported off pybind
  src/planning/     occupancy grid, ESDF, trajectory library (the 6 njit fns)
  src/mapping/      VLAD, fusion window, A*, path priors
  src/trt/          TensorRT C++ wrappers (8 engines + CUDA Graph)
  src/components/   perception / mapping / planning / imu_propagator
                    rclcpp Components + src/main.cpp single-process host
fixtures/           golden I/O for alignment testing (gitignored)
docs/               design docs
```

## Build & run (in the tinynav container)

```bash
docker run --rm --gpus all --network host \
  -v "$PWD":/ws -w /ws uniflexai/tinynav:latest bash -c \
  'source /opt/ros/humble/setup.bash && colcon build --packages-select tinynav_cpp'

# run the whole stack (single process, 4 components):
docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \
  uniflexai/tinynav:latest bash -c \
  'source /opt/ros/humble/setup.bash && source install/setup.bash && \
   ros2 launch tinynav_cpp tinynav.launch.py'
```

TensorRT engines are expected at `/tinynav/tinynav/models/*.plan` (present in
the image); `model_dir` is a parameter.

## Known gaps (v1)

- Perception uses PnP + IMU propagation; the GTSAM factor-graph refinement is
  behind an interface and lands with the docker GTSAM layer (image has no
  libgtsam-dev).
- Map persistence moves from shelve/pickle to a portable "map format v2"
  (binary blobs + JSON meta) — the Python exporter is pending.
