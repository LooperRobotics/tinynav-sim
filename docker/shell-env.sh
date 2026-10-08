#!/usr/bin/env bash

# Shared environment for the service entrypoint and every shell opened later
# with `docker exec`. Keep this file safe to source more than once.
source /opt/ros/humble/setup.bash

if [[ -d ${XDG_RUNTIME_DIR:-/nonexistent} ]]; then
  if [[ -z ${XAUTHORITY:-} && -f "$XDG_RUNTIME_DIR/gdm/Xauthority" ]]; then
    export XAUTHORITY="$XDG_RUNTIME_DIR/gdm/Xauthority"
  fi
  if [[ -z ${XAUTHORITY:-} ]]; then
    for xauth_file in "$XDG_RUNTIME_DIR"/.mutter-Xwaylandauth.*; do
      if [[ -f "$xauth_file" ]]; then
        export XAUTHORITY="$xauth_file"
        break
      fi
    done
    unset xauth_file
  fi
fi

if [[ -z ${DISPLAY:-} ]]; then
  if [[ ${XAUTHORITY:-} == */gdm/Xauthority && -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
  elif [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
  elif [[ -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
  fi
fi

tinynav_install=/workspace/tinynav-sim/.container/install/setup.bash
export TINYNAV_INSTALL_PREFIX=/workspace/tinynav-sim/.container/install
if [[ -f "$tinynav_install" ]]; then
  source "$tinynav_install"
fi

# DDS wiring so this shell can talk to the local mjsim (hil) container:
# CycloneDDS pinned to loopback unicast, both halves required -- a Fast DDS
# peer drops the stereo frames to ~4 Hz, and an unpinned one cannot even
# discover the pinned publisher. Non-clobbering, so a split-site launch that
# exports its own CYCLONEDDS_URI still wins (see the script's header).
_hil_peer_env="$(dirname "${BASH_SOURCE[0]}")/../tools/probes/hil_peer_env.sh"
if [[ -f "$_hil_peer_env" ]]; then
  source "$_hil_peer_env"
fi
unset _hil_peer_env

unset tinynav_install
