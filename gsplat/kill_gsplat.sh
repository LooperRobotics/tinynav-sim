#!/usr/bin/env bash
# Fully stop every gsplat rig process in this container. run_gsplat.sh runs
# this first (relaunch semantics); it is also the manual "stop the rig" tool.
# Mirrors gazebo/kill_sim.sh (TERM then KILL, tmux session, shm cleanup).
#
# Safe against the pkill self-match trap: the patterns are split so they never
# appear in this script's own argv (it runs as `bash .../kill_gsplat.sh`) --
# and never put the bare names in the SAME docker exec command line elsewhere.
PATTERNS=(
  'sensor''_server.py'
  'gs_ros''_bridge.py'
  'view''_window.py'
  'tinynav''_node'
  'ros2 launch tinynav''_cpp'
  'perception''_node'
  'planning''_node'
  'simulator''_control'
  'keyboard''_teleop'
  'ros2 topic pub'
  'ros2 service call'
)

for p in "${PATTERNS[@]}"; do pkill -TERM -f "$p" 2>/dev/null; done
sleep 2
for p in "${PATTERNS[@]}"; do pkill -KILL -f "$p" 2>/dev/null; done
tmux kill-session -t tinynav_gs 2>/dev/null
rm -f /dev/shm/gsplay_sensors.bin /dev/shm/gsplay_cmd.txt /dev/shm/gsplay_cmd.txt.tmp \
      /dev/shm/gsplay_state.bin

# FastDDS shared-memory segments whose owner process is gone (same reason as
# gazebo/kill_sim.sh): leftover segments slow the next DDS init.
for f in /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; do
  [ -e "$f" ] || continue
  if ! grep -qs "$f" /proc/*/maps 2>/dev/null; then
    rm -f "$f"
  fi
done

sleep 1
LEFT=0
for p in "${PATTERNS[@]}"; do
  # zombies (PID 1 never reaps here) must not count as alive — they hold no
  # GPU/memory/ports, only a process-table entry
  if pgrep -f "$p" 2>/dev/null | xargs -r ps -o stat= -p 2>/dev/null | grep -qv '^Z'; then
    echo "still alive: $p"
    LEFT=1
  fi
done
if tmux has-session -t tinynav_gs 2>/dev/null; then
  echo "tmux session tinynav_gs still alive"
  LEFT=1
fi
if [ "$LEFT" = 0 ]; then
  echo "gsplat rig fully stopped (server, bridge, stack, tmux, shm ring)"
fi
exit "$LEFT"
