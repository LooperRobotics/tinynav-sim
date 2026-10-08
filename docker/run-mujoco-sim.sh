#!/bin/bash
# Run the overall-sim container with everything the sim needs:
#   GPU (Vulkan/EGL/GLX) + X11 (viewer window) + /dev/input (keyboard) +
#   host network (DDS domain shared with the tinynav container / navcore).
#
# Default form is hil: DDS contract topics + viser web viewer :8012 +
# keyboard teleop (arrows override /cmd_vel). sim = the no-DDS sandbox.
#
# Usage: bash docker/run-mujoco-sim.sh [hil|sim|mj] [extra entrypoint args]
#       e.g. bash docker/run-mujoco-sim.sh sim --web-port 8014
set -u
FORM="${1:-hil}"; shift || true
NAME="mjsim-${FORM}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# keyboard capture needs the input group's gid from the host
#
# /dev/input is bind-mounted live WITH a device-cgroup rule for the whole
# input major (13) rather than passed as `--device /dev/input`: --device
# snapshots the nodes present at container creation, so a keyboard that
# appears later (re-plugged, re-paired, or simply a different event number)
# stays invisible until the container is restarted -- which reads as
# "the keyboard stopped working" while nothing in the sim changed.
INPUT_GID="$(getent group input | cut -d: -f3)"

docker run -d --name "$NAME" \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY="${DISPLAY:-:1}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ${XDG_RUNTIME_DIR:+-v "$XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR" -e XDG_RUNTIME_DIR} \
  -v /dev/input:/dev/input \
  --device-cgroup-rule "c 13:* rwm" \
  ${INPUT_GID:+--group-add "$INPUT_GID"} \
  $(for d in /dev/dri/card0 /dev/dri/card1 /dev/dri/renderD128 /dev/dri/renderD129; do [ -e "$d" ] && echo -n "--device $d "; done) \
  --network host \
  --ipc host \
  -e MUJOCO_GL=egl \
  mjsim:jazzy "$FORM" "$@"

echo "container '$NAME' up (form=$FORM)"
echo "  logs:   docker logs -f $NAME"
echo "  web:    http://127.0.0.1:8012/   (viser viewer; --web-port N moves it)"
echo "  stream: http://127.0.0.1:8888/   (--preview only)"
echo "  topics: /camera/camera/infra1|infra2|depth|color|vio_*|imu  (DDS)"
echo "  stop:   docker rm -f $NAME"
