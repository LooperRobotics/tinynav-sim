#!/usr/bin/env bash
# DDS preflight: refuse to start into a known-dead DDS environment.
#
# Why: clash-verge/mihomo in TUN mode hijacks every non-loopback route
# (`ip rule` entry "lookup 2022" from the 198.18.0.0/30 Meta interface) —
# DDS discovery/multicast on physical NICs then dies silently (subscribers
# see 0 frames, publishers block per-frame), and it looks exactly like a
# broken stack. Loopback is EXEMPT (the rule says "not from all iif lo"), so
# the single-host loopback-unicast Cyclone config
# (tools/probes/cyclone_localhost_unicast.xml, baked into the pure-sim image
# at /opt/dds/) keeps working — see docs/plan-sim-image-split.md §5.
#
# Called by gazebo/run_simulator.sh and gsplat/run_gsplat.sh after their DDS
# env is set up. Exit 1 only on the fatal combination: TUN active AND no
# loopback-unicast URI.
set -u

if ! ip rule show 2>/dev/null | grep -q "lookup 2022"; then
  exit 0    # no TUN signature: nothing to say
fi

case "${CYCLONEDDS_URI:-}" in
  *localhost_unicast*)
    echo "[dds-preflight] TUN mode detected — loopback-unicast Cyclone config active, OK"
    exit 0 ;;
esac

echo "[dds-preflight] ERROR: clash TUN mode is hijacking non-loopback routes;" >&2
echo "  DDS on physical NICs will silently die (discovery black-holed)." >&2
echo "  Pick one:" >&2
echo "    - single host: run with the loopback-unicast config" >&2
echo "      CYCLONEDDS_URI=file:///opt/dds/cyclone_localhost_unicast.xml  (sim image)" >&2
echo "      (or file://<repo>/tools/probes/cyclone_localhost_unicast.xml elsewhere)" >&2
echo "    - split-site / USB link: turn off TUN for the sim session" >&2
exit 1
