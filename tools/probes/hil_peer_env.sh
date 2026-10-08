#!/bin/bash
# Local DDS wiring for the tinynav (peer) container, so its stack / rqt /
# ros2 CLI interoperate with the mjsim hil contract face on the same host.
#
#   source <repo>/tools/probes/hil_peer_env.sh
#
# Wired in by default: docker/shell-env.sh sources this (every shell opened
# with `docker exec` on the rig, plus the service entrypoint), so nothing has
# to be remembered. Re-sourcing is harmless.
#
# BOTH settings are required -- each alone is measurably broken (verified):
#   RMW_IMPLEMENTATION  mjsim speaks CycloneDDS. A Fast DDS subscriber
#     (Humble's default while this is unset) DOES interoperate, but drops
#     most large best-effort frames: 544x640 stereo images fell 10 Hz -> 3.9.
#   CYCLONEDDS_URI      mjsim's Cyclone is pinned to loopback unicast with
#     multicast off (that is what keeps big frames off the VPN TUN). A
#     default-config peer binds the default-route interface and never meets
#     it -- 0 Hz, discovery included.
#
# Non-clobbering by design: an already-set-and-valid value is left alone, so
# a split-site launch that exports its own CYCLONEDDS_URI / RMW still wins
# (gazebo/launch/sim.launch.py sets its unicast-peer config explicitly).
# Only an unset var, or a URI pointing at a file that does not exist, is
# replaced -- that last case is the rig image's baked path when the file is
# missing from the container.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# candidate configs, in preference order: the rig image's baked path first
# (same file, kept in sync by that image), then the repo copy next to us.
_hil_cfg=""
for _c in /tinynav/scripts/cyclone_dds_localhost.xml \
          "$_here/cyclone_localhost_unicast.xml"; do
  if [[ -f $_c ]]; then _hil_cfg=$_c; break; fi
done
unset _c

if [[ -n ${CYCLONEDDS_URI:-} ]]; then
  _cur=${CYCLONEDDS_URI#file://}
  if [[ ! -f $_cur && -n $_hil_cfg ]]; then
    export CYCLONEDDS_URI="file://$_hil_cfg"
    echo "[hil_peer_env] CYCLONEDDS_URI pointed at a missing file -> $_hil_cfg"
  fi
elif [[ -n $_hil_cfg ]]; then
  export CYCLONEDDS_URI="file://$_hil_cfg"
fi
unset _cur

if [[ -z ${RMW_IMPLEMENTATION:-} ]]; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi
unset _hil_cfg

# quiet by default (sourced from every interactive shell); set
# HIL_PEER_ENV_VERBOSE=1 to see the resolved values.
if [[ -n ${HIL_PEER_ENV_VERBOSE:-} ]]; then
  echo "[hil_peer_env] RMW=$RMW_IMPLEMENTATION CYCLONEDDS_URI=${CYCLONEDDS_URI:-unset}"
fi
