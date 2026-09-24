#!/bin/bash
# gsplat (3DGS) simulator launcher: one named tmux window per component.
# Mirrors gazebo/run_simulator.sh; the "sim face" is the MotrixSim + gsplat
# sensor server + ROS bridge instead of gz.
#
# Usage:  bash gsplat/run_gsplat.sh [--stack full|sensor|cpp] [--map --map-dir <v2 dir>]
#                                  [--scene church|nav1|<config name>] [--rtf <f>] [--db <path>]
#                                  [-- extra sensor_server args, e.g. --drive-distance 12]
#
# Windows (tmux session tinynav_gs; rviz on by default, the MotrixSim viewer
# (--gui) off by default — it renders through Mesa llvmpipe and measurably
# steals CPU from the camera stream, so it is opt-in):
#   gssim    the 3DGS sensor server  (/opt/venv_gs python: physics + RL policy + render)
#   bridge   ring -> /camera/camera/... + /camera/camera/imu + /sim/gt_pose + /clock
#   control  trajectory follower (planning trajectory -> /cmd_vel; same node as gz rigs)
#   percept/planning[/map]  python reference stack   (--stack full)
#   cpp      single-process C++ stack                 (--stack cpp)
#   teleop / rviz as available
#
# --stack sensor: only the sensor face, for when pilot (or another consumer)
#          owns the rest. Conflicts with --map.
# --map:   only with --stack cpp; loads a map format v2 directory and enables
#          keyframe relocalization + global planning on the C++ side. WITHOUT
#          --map the cpp stack runs in BUILD mode: drive with teleop (or a
#          --drive-* task) and call /mapping/start + /mapping/stop to produce
#          a map directly (see docs/sim-mapping-runbook.md 4.4).
#
# Env: GS_SERVER_ARGS adds args to sensor_server (same as trailing `-- ...`);
#      GS_VENV (default /opt/venv_gs), GS_CUDA_HOME (default /opt/cuda-shim-gs).
#
# Attach:  tmux attach -t tinynav_gs

SESSION=tinynav_gs
GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(dirname "$GS_ROOT")"
# Reference snapshot first on PYTHONPATH (namespace-package trick, see
# gazebo/README.md); the image python3 already carries site-packages + gtsam.
export PYTHONPATH="$WS_ROOT/reference:${PYTHONPATH}"
# CycloneDDS everywhere, and clear the image's baked CYCLONEDDS_URI that points
# at a file missing in this container (same reason as gazebo/run_simulator.sh;
# for the bridge it is not just cosmetic: Fast DDS drops the 783 KB color image
# to 2.5 Hz).
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI=""

GS_VENV="${GS_VENV:-/opt/venv_gs}"
GS_CUDA_HOME="${GS_CUDA_HOME:-/opt/cuda-shim-gs}"

STACK=full
WITH_MAP=0
SCENE=church
RTF=1
GUI=0
RVIZ=1        # rviz starts by default (like gazebo/run_simulator.sh); --no-rviz to drop it
DB_PATH=""
MAP_DIR_OVERRIDE=""
SERVER_EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --stack) STACK="$2"; shift 2 ;;
    --map) WITH_MAP=1; shift ;;
    --map-dir) MAP_DIR_OVERRIDE="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --rtf) RTF="$2"; shift 2 ;;
    --gui) GUI=1; shift ;;          # decoupled viewer window (opt-in)
    --gui-embed) GUI=2; shift ;;    # legacy in-process viewer (slower sim)
    --rviz) RVIZ=1; shift ;;        # (default) rviz window
    --no-rviz) RVIZ=0; shift ;;     # drop rviz when you want every bit of rtf
    --db) DB_PATH="$2"; shift 2 ;;
    --) shift; SERVER_EXTRA=("$@"); break ;;
    *) echo "usage: bash $0 [--stack full|sensor|cpp] [--map --map-dir <dir>] [--scene church|nav1] [--rtf <f>] [--gui|--gui-embed] [--no-rviz] [--db <path>] [-- <sensor_server args>]"; exit 1 ;;
  esac
done
# Viewer: --gui runs a DECOUPLED window process (gsplat/tools/view_window.py)
# reading the state slot the server writes — measured rtf 0.92 with a 25-30 fps
# window, vs rtf 0.68 / 15 fps when the viewport renders inside the server
# (--gui-embed; kept for single-process debugging). The container's wgpu only
# has Mesa llvmpipe (no GPU GL/Vulkan), so the viewport costs real CPU time —
# keeping it out of the sim process is the whole point.
if [[ $GUI == 2 ]]; then
  SERVER_EXTRA+=(--window --window-fps 15 --window-panels off)
fi
SERVER_EXTRA+=(${GS_SERVER_ARGS:-})
[[ $STACK != full && $STACK != sensor && $STACK != cpp ]] && { echo "--stack must be 'full', 'sensor' or 'cpp'"; exit 1; }
if [[ $STACK == sensor && $WITH_MAP == 1 ]]; then
  echo "--stack sensor conflicts with --map: the consumer owns localization"; exit 1
fi
if [[ $WITH_MAP == 1 && $STACK == full ]]; then
  echo "--map is only wired for --stack cpp (map format v2) here; the python map_node path is gazebo-only"; exit 1
fi
[[ ! -x $GS_VENV/bin/python ]] && { echo "gs venv missing: $GS_VENV (run gsplat/setup_env.sh)"; exit 1; }
[[ ! -d $GS_CUDA_HOME ]] && { echo "CUDA toolkit shim missing: $GS_CUDA_HOME (run gsplat/setup_env.sh)"; exit 1; }
if [[ $STACK == cpp && ! -f $WS_ROOT/install/setup.bash ]]; then
  echo "--stack cpp needs $WS_ROOT/install/setup.bash -- colcon build first (see gazebo/README.md)"; exit 1
fi
[[ -n $DB_PATH ]] && export TINYNAV_DB_PATH="$DB_PATH"

# relaunch semantics: previous rig fully gone before this one starts
bash "$GS_ROOT/kill_gsplat.sh" || true

MAP_DIR="${MAP_DIR_OVERRIDE:-$WS_ROOT/output/map_gsplat}"
cd "$WS_ROOT"
mkdir -p logs

command -v ros2 >/dev/null 2>&1 || source /opt/ros/humble/setup.bash
[[ -f /3rdparty/message_filters_ws/install/local_setup.bash ]] && source /3rdparty/message_filters_ws/install/local_setup.bash

win() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -d -t "$SESSION" -n "$1" -c "$WS_ROOT" "exec bash -i"
  else
    tmux new-session -d -s "$SESSION" -n "$1" -c "$WS_ROOT" "exec bash -i"
  fi
  tmux set-option -w -t "$SESSION:$1" remain-on-exit on
  tmux set-option -w -t "$SESSION:$1" automatic-rename off
  tmux send-keys -t "$SESSION:$1" "export PYTHONPATH=\"$PYTHONPATH\" TINYNAV_DB_PATH=\"$TINYNAV_DB_PATH\" RMW_IMPLEMENTATION=\"$RMW_IMPLEMENTATION\" CYCLONEDDS_URI=\"\"" Enter
  tmux send-keys -t "$SESSION:$1" "$2" Enter
}

# ---- sensor face -----------------------------------------------------------
# The server keeps CUDA_HOME/PATH because gsplat checks for nvcc at import and
# refuses to load its compiled kernels without it (they are already JIT-compiled
# into the torch-extensions cache -- no compile happens at runtime).
#
# --gui note: /root/.bashrc bakes the NVIDIA PRIME/EGL trio for rviz/gz (good
# for them, kept on their windows), but wgpu -- the MotrixSim viewer -- probes
# NVIDIA EGL under it and panics. The server drops the trio itself in --window
# mode (before importing motrixsim.render), so any launcher path is safe; the
# window here only adds WAYLAND_DISPLAY plus a writable XDG_RUNTIME_DIR.
# (They must be exported in THIS window's shell, hence the %%s splice below.)
if [[ $GUI == 1 ]]; then
  GSSIM_GUI_ENV="unset __GLX_VENDOR_LIBRARY_NAME __NV_PRIME_RENDER_OFFLOAD __EGL_VENDOR_LIBRARY_FILENAMES; export WAYLAND_DISPLAY= XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp};"
else
  GSSIM_GUI_ENV=""
fi
win gssim "$GSSIM_GUI_ENV export CUDA_HOME=$GS_CUDA_HOME PATH=\"$GS_CUDA_HOME/bin:\$PATH\" TORCH_CUDA_ARCH_LIST=12.0 GS_PLAYGROUND_ROOT=${GS_PLAYGROUND_ROOT:-/workspace/github/simulation/gs_playground}; $GS_VENV/bin/python -u $GS_ROOT/server/sensor_server.py --config $SCENE --rtf $RTF ${SERVER_EXTRA[*]} 2>&1 | tee logs/gssim.log"

# wait for the ring before the bridge (bridge retries anyway, but a clean
# start keeps the first camera frame from being missed by recorders)
RING=/dev/shm/gsplay_sensors.bin
echo "waiting for sensor ring $RING ..."
for _ in $(seq 1 120); do
  [[ -f $RING ]] && head -c 4 "$RING" | grep -q GSPG && break
  sleep 1
done
[[ -f $RING ]] || { echo "sensor ring never appeared -- check the gssim window"; exit 1; }

win bridge "python3 -u $GS_ROOT/ros/gs_ros_bridge.py --report-every 15 2>&1 | tee logs/gsbridge.log"

if [[ $GUI == 1 ]]; then
  # decoupled viewer: same scene as the server, driven by the state slot
  # `env -i`: the interactive window shell sources /root/.bashrc, whose
  # NVIDIA PRIME/EGL trio (fine for rviz/gz) makes the viewer's wgpu path
  # stall (~1 s per frame, measured); a minimal env is the reliable form.
  win viewer "env -i DISPLAY=${DISPLAY:-:1} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp} HOME=$HOME PATH=/usr/bin:/bin GS_PLAYGROUND_ROOT=${GS_PLAYGROUND_ROOT:-/workspace/github/simulation/gs_playground} $GS_VENV/bin/python -u $GS_ROOT/tools/view_window.py --scene $SCENE --fps 30 2>&1 | tee logs/gsview.log"
fi

# ---- stack -----------------------------------------------------------------
if [[ $STACK != cpp ]]; then
  if [[ $STACK == full ]]; then
    # python reference stack: perception + planning, raw-stream remap not needed
    # (no sim_gt_reloc on the gsplat side yet)
    win percept "python3 $WS_ROOT/reference/tinynav/core/perception_node.py 2>&1 | tee logs/percept.log"
  fi
else
  CPP_LAUNCH_ARGS=""
  [[ $WITH_MAP == 1 ]] && CPP_LAUNCH_ARGS="map_path:=$MAP_DIR"
  # single-process C++ stack: perception + mapping + planning + imu_propagator
  win cpp "source $WS_ROOT/install/setup.bash && ros2 launch tinynav_cpp tinynav.launch.py $CPP_LAUNCH_ARGS 2>&1 | tee logs/cpp.log"
fi

# trajectory follower on every rig: planning publishes /planning/trajectory_path,
# this node turns it into /cmd_vel -> cmd file -> RL policy
win control "python3 $WS_ROOT/reference/tinynav/platforms/simulator_control.py 2>&1 | tee logs/control.log"

if [[ $STACK == full ]]; then
  PLAN_ARGS="-p min_wall_span_m:=0.2"
  win planning "python3 $WS_ROOT/reference/tinynav/core/planning_node.py $PLAN_ARGS 2>&1 | tee logs/planning.log"
fi

HAVE_PYNPUT=1
python3 -c "import pynput" 2>/dev/null || HAVE_PYNPUT=0
if [[ $HAVE_PYNPUT == 1 ]]; then
  win teleop "python3 $WS_ROOT/reference/tinynav/platforms/keyboard_teleop.py 2>&1 | tee logs/teleop.log"
else
  echo "WARN: pynput not installed -- teleop window skipped"
fi
# rviz starts by default (same as gazebo/run_simulator.sh). It renders through
# Mesa llvmpipe like the MotrixSim viewer, so it does cost CPU the camera
# stream would otherwise use -- drop it with --no-rviz when you want the last
# bit of rtf. The MotrixSim viewer (--gui) stays opt-in: it is the heavier one.
if [[ $RVIZ == 1 ]]; then
  win rviz "rviz2 -d /tinynav/docs/vis.rviz 2>&1 | tee logs/rviz.log"
fi

echo "session '$SESSION' up (STACK=$STACK, SCENE=$SCENE, WITH_MAP=$WITH_MAP, RTF=$RTF, GUI=$GUI{1=decoupled viewer,2=embedded}, RVIZ=$RVIZ):"
tmux list-windows -t "$SESSION" -F '  #{window_index}:#{window_name}'
echo "attach: docker exec -t tinynav tmux attach -t $SESSION   (Ctrl+B D detach)"
echo "state:  bash $GS_ROOT/gs_state.sh --slam"
