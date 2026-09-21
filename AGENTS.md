# AGENTS.md — tinynav-sim

## What this repo is

The C++ rewrite of TinyNav's hot-path nodes, developed against the gzsim
simulation as the end-to-end test bench. Replaces perception / mapping /
planning / imu_propagator with rclcpp components running in ONE process with
intra-process (zero-copy) communication.

## Hard rules

- `reference/` is a **read-only snapshot** of tinynav branch `gzy/fix/gzsim`
  (core py + cpp kernels). It is the spec: when C++ and Python disagree, the
  Python semantics win. Never edit files under `reference/`.
  - `reference/tinynav/` is a namespace package on purpose — do NOT add
    `__init__.py`, it would shadow the image's `tinynav.tinynav_cpp_bind`
    (.so needed by map_node). `reference/tinynav/platforms/` was added per
    direct user request (sim stack needs simulator_control/keyboard_teleop).
- `src/tinynav_cpp/src/core|kernels|planning|mapping|trt` must NOT include ROS
  headers — they are pure libraries, gtest-able without a ROS graph.
  ROS lives only in `src/components/`.
- Ported functions keep the Python name (snake_case) and a top-of-file comment
  `// Port of reference/tinynav/core/<file>::<func>` so alignment review can
  walk function-by-function.
- Numerical fidelity: double everywhere Python used float64; no fast-math.
- `fixtures/` never enters git.

## Conventions

- C++17, Eigen3 for linear algebra, OpenCV for images/PnP, TRT 10 C++ API
  (enqueueV3 / setTensorAddress — the image ships TRT 10).
- Components: one class per node domain, default MutuallyExclusive callback
  group (matches Python single-threaded semantics); publish via
  `std::unique_ptr<Message>`, subscribe with `ConstSharedPtr` callbacks, and
  `NodeOptions().use_intra_process_comms(true)`.
- Engine load failure must degrade (log + disable the path), never crash the
  process — nodes must run on machines without models/GPU.
- Build/verify only inside the container (`uniflexai/tinynav:latest`), never
  on the host. See README.md for the commands.

## Environment facts (image `uniflexai/tinynav:latest`)

- ROS2 Humble, g++ 11.4, cmake 3.22, OpenCV 4.5.4, Eigen3, yaml-cpp,
  TensorRT 10 headers at `/usr/include/x86_64-linux-gnu/NvInfer.h`,
  ignition-fortress 6.17.1, **no libgtsam-dev**.
- Python venv site-packages: `/opt/venv/lib/python3.10/site-packages` (the
  image's `python3` IS the venv python; PYTHONPATH already carries it plus
  `/3rdparty/gtsam/build/python`).
- TRT engines: `/tinynav/tinynav/models/*.plan`.

## Rig container quick facts (instance `tinynav`, image tinynav-runtime:x86_64)

The user's GPU rig (full go2 gait chain, ros_gz_sim, .plan engines) — where
every sim/e2e verification runs. Container name is `tinynav`; the repo mounts
at `/workspace/dm/tinynav-sim` (NOT /ws). Gotchas:

- colcon build must be re-run after switching containers (CMake cache has
  stale absolute paths): `source /opt/ros/humble/setup.bash && colcon build
  --packages-select tinynav_cpp`.
- `export PYTHONPATH=...` must come BEFORE `source /opt/ros/humble/setup.bash`
  (reverse order wipes the ROS site-packages from PYTHONPATH).
- Plain `python3` here is the system python (no venv); the tinynav venv is
  `/opt/venv/bin/python3`. `python3 -u` when piping (block buffering hides
  progress and once masked reloc successes).
- shelve opens a path WITHOUT the `.db` suffix (libdb ndbm compat layer).
- Sourcing ROS setup.bash under `set -u` exits the script (unbound vars) —
  `set +u` first.
- Long C++ snippets: write on the host and `docker cp`; heredocs inside a
  double-quoted docker exec string eat the `$`s.

## Debug & ops toolbox (all verified 2026-09-19)

- `sim/dog_state.sh [--slam] [--map-dir <v2 map dir>]` — one-shot robot state:
  gz ground truth + yaw in degrees, SLAM odom (odom frame is offset ~-90° from
  gz world; never compare the two yaws directly), and an IN/OUT verdict against
  the mapped-trajectory bounding box (margin 0.5 m). Run BEFORE every new sim
  operation: off-trajectory reloc failures are expected behavior, not bugs.
- Reloc failure dump channel: start the stack with
  `TINYNAV_RELOC_DUMP_DIR=<dir>` and the mapping component writes the exact
  LightGlue inputs of every rejected reloc candidate to
  `<dir>/<live_ts>_<cand_ts>/` (6 f32 files + meta.json + live.png, cap 50
  pairs). Replay with `tools/probes/probe_lg <dump dir>` — the @848 match
  count must equal meta.json's match_count (bit parity with the component).
- `tools/probes/` — standalone TRT probes (git-tracked); `bash
  tools/probes/build.sh` builds them against the colcon tree. probe data /
  dumps live in gitignored `fixtures/`.
- `tools/grab_topic.py <topic> --count N [--out DIR]` — one-shot topic grabber
  (serialized .bin for any type, decoded .npy for Image). Use instead of
  `ros2 topic echo --field data --raw`, which returns 0 bytes on some topics.
- Per-node logs: the stack writes `<TINYNAV_DB_PATH>/logs/<YYYY-MM-DD>/<tag>.log`
  (tags: perception / map / planning / imu_propagator / tinynav + console.log)
  in the SAME tree as the python stack, via a rcutils hook — RCLCPP_* call
  sites need no changes. DEBUG+ in files, INFO+ on console; 10MB size fuse
  (`TINYNAV_LOG_MAX_MB`, 0=off); 14-day retention. Node names are the ported
  ones (no launch name override) — `ros2 param get /planning_node ...` works.

## Split-site rehearsal facts (2026-09-20)

- Layout: container A (tinynav) runs `sim/launch/sim.launch.py` (owns the
  FastDDS discovery server, port 11811) + `tinynav_cpp/launch/
  perception.launch.py` (perception component, /slam/* remapped to
  /camera/camera/slam/*); container B (tinynav-orin) runs `tinynav_cpp/
  launch/orin_stack.launch.py` (looper_bridge relay + imu/mapping/planning,
  use_sim_time). Both need `TINYNAV_DB_PATH` for the per-node logs.
- ANY ros2 CLI diagnostic on these sites needs the site env:
  `export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 ROS_DISCOVERY_SERVER=127.0.0.1:11811`
  — AND the CLI is still BLIND for graph/data queries under the discovery
  server (Humble ros2cli): `ros2 node list` / `topic hz` report nothing
  while real nodes exchange messages. Verify with
  `python3 tools/probes/probe_first_msg.py <topic> <Image|Odometry|Imu|...>`,
  never conclude "link broken" from CLI output.
- Cleanup order: kill_sim.sh (alone in its own command), then
  `pkill -f fast-discovery-server` — the discovery server's C++ child
  survives its python wrapper and holds port 11811 (later launches fail
  with "wasn't able to allocate the specified listening port"). Zombie
  processes (PID 1 never reaps) make process tables lie; trust the data
  plane and logs. `pkill -f <pattern>` matches your own docker exec command
  line — it has killed debugging shells repeatedly (exit 137/143).
