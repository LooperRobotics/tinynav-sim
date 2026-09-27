#!/usr/bin/env bash

# Shared environment for the service entrypoint and every shell opened later
# with `docker exec`. Keep this file safe to source more than once.
source /opt/ros/humble/setup.bash

if [[ -z ${XAUTHORITY:-} && -d ${XDG_RUNTIME_DIR:-/nonexistent} ]]; then
  for xauth_file in "$XDG_RUNTIME_DIR"/.mutter-Xwaylandauth.*; do
    if [[ -f "$xauth_file" ]]; then
      export XAUTHORITY="$xauth_file"
      break
    fi
  done
  unset xauth_file
fi

tinynav_install=/workspace/tinynav-sim/.container/install/setup.bash
export TINYNAV_INSTALL_PREFIX=/workspace/tinynav-sim/.container/install
if [[ -f "$tinynav_install" ]]; then
  source "$tinynav_install"
fi

unset tinynav_install
