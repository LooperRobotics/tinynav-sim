#!/usr/bin/env bash
# 停 HIL(both x86 windows)。回真机:navcore 侧 bash ~/navcore-deploy/pilot/hil_off.sh
tmux kill-session -t tinynav_hil 2>/dev/null && echo "tmux session tinynav_hil killed" || echo "no tinynav_hil session"
pkill -f "ros/looper_emu.py" 2>/dev/null && echo "looper_emu stopped"
pkill -f "server/sensor_server.py" 2>/dev/null && echo "sensor_server stopped"
exit 0
