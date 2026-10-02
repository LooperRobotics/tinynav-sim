#!/usr/bin/env bash
# Watchdog for the decoupled MotrixSim viewer (--gui).
#
# Why: once the viewer's wgpu presentation path degrades mid-run (measured
# viewer reports collapse to 1.0 Hz with ~1000 ms sync and NEVER
# recovers in-process — killing the sim does not help either), the window is
# useless until the process is restarted. The viewer is stateless (it only
# reads /dev/shm/gsplay_state.bin), so a restart is cheap and safe.
#
# Env (exported by run_gsplat.sh): WATCH_SESSION (tmux session),
# WATCH_VIEWER_CMD (the exact command line `win viewer` sent), WATCH_WS_ROOT.
# Logs to logs/view_watchdog.log (appended by run_gsplat.sh).
set -u
SESSION="${WATCH_SESSION:?}"
CMD="${WATCH_VIEWER_CMD:?}"
WS_ROOT="${WATCH_WS_ROOT:?}"
WINDOW=viewer
LOG="$WS_ROOT/logs/gsview.log"
LOW=0

viewer_window_up() {
  tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$WINDOW"
}

respawn_viewer() {
  echo "$(date +%T) respawning viewer (LOW=$LOW)"
  tmux kill-window -t "$SESSION:$WINDOW" 2>/dev/null
  sleep 2
  tmux new-window -d -t "$SESSION" -n "$WINDOW" -c "$WS_ROOT" "exec bash -i"
  tmux set-option -w -t "$SESSION:$WINDOW" remain-on-exit on
  tmux set-option -w -t "$SESSION:$WINDOW" automatic-rename off
  tmux send-keys -t "$SESSION:$WINDOW" "$CMD" Enter
  LOW=0
}

while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep 10
  if ! viewer_window_up; then
    echo "$(date +%T) viewer window gone — respawning"
    respawn_viewer
    continue
  fi
  hz=$(grep -oE 'viewer [0-9.]+ Hz' "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  if [[ -n "$hz" ]] && awk -v h="$hz" 'BEGIN{exit !(h < 5.0)}'; then
    LOW=$((LOW + 1))
    echo "$(date +%T) viewer slow: ${hz} Hz (LOW=$LOW)"
  else
    LOW=0
  fi
  if [ "$LOW" -ge 3 ]; then
    respawn_viewer
  fi
done
echo "$(date +%T) session $SESSION gone — watchdog exiting"
