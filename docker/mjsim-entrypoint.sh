#!/bin/bash
# Entrypoint for the overall-sim container: run the sim entry DIRECTLY (no
# tmux inside containers). The hil form publishes DDS straight into the
# shared host network domain (rclpy) -- that is the whole point of this
# image; there is no transport switch anymore.
set -e
FORM="${1:-hil}"
shift || true
ROOT=/workspace/tinynav-sim/mujoco
cd "$ROOT"
# rclpy lives in the ROS prefix (PYTHONPATH + AMENT_PREFIX_PATH), not on the
# default python path -- the venv only shadows pip packages
source /opt/ros/jazzy/setup.bash
export MUJOCO_GL=egl
export DISPLAY="${DISPLAY:-:1}"

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "$GPU" in
  *5070*) DEV="5070" ;;
  *2060*) DEV="2060" ;;
  *)      DEV="5070" ;;
esac

ENTRY="view.py"
[ "$FORM" = "mj" ] && ENTRY="physics_view.py"
[ "$FORM" = "hil" ] && ENTRY="hil.py"

# --spawn comes from the scene registry default (plant.SCENES); pass
# --spawn/--scene explicitly to override.
exec python -u "$ENTRY" "$DEV" "$@"
