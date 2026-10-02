#!/usr/bin/env bash
# HIL 启动:x86 侧 3DGS 仿真冒充 looper 相机盒(路线 B)。
# 文档: docs/hil-looper-emu-plan.md, 契约: docs/looper-contract.md
#
# 前置:
#   1. navcore(012) 已 hil_on: bash ~/navcore-deploy/pilot/hil_on.sh
#      (core_runtime 改绑 l4tbr0 + 白名单,真 looper 已隔离)
#   2. 本机 USB 网卡是 192.168.55.100(enxae0e58315f4c,非此 IP 改下面的 XML)
#
# 与 run_gsplat.sh --stack sensor 的区别:这里跑 looper_emu.py 而不是
# gs_ros_bridge.py —— 两者都发 /camera/camera/infra1/*,同跑会双源打架、
# ExactTime 全废。emu 直读共享内存 ring,无 DDS 中转。
#
# Usage:  bash gsplat/run_hil.sh [--scene church|nav1|map2] [--rtf 1]
#                                    [--cam-hz 12] [-- <sensor_server 额外参数>]
# Attach: tmux attach -t tinynav_hil    # 窗口: gssim / emu
# 停止:   bash gsplat/kill_hil.sh(或 tmux kill-session -t tinynav_hil)
# 收工:   navcore 侧 bash hil_off.sh 回真机。

set -uo pipefail

SESSION=tinynav_hil
GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(dirname "$GS_ROOT")"

GS_VENV="${GS_VENV:-/opt/venv_gs}"
GS_CUDA_HOME="${GS_CUDA_HOME:-/opt/cuda-shim-gs}"

SCENE=church
RTF=1
CAM_HZ=12
SERVER_EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --scene) SCENE="$2"; shift 2 ;;
    --rtf) RTF="$2"; shift 2 ;;
    --cam-hz) CAM_HZ="$2"; shift 2 ;;
    --) shift; SERVER_EXTRA=("$@"); break ;;
    *) echo "usage: bash $0 [--scene church|nav1|map2] [--rtf <f>] [--cam-hz 12] [-- <sensor_server args>]" >&2; exit 1 ;;
  esac
done

[[ ! -x $GS_VENV/bin/python ]] && { echo "gs venv missing: $GS_VENV (run gsplat/setup_env.sh)"; exit 1; }
[[ ! -d $GS_CUDA_HOME ]] && { echo "CUDA toolkit shim missing: $GS_CUDA_HOME (run gsplat/setup_env.sh)"; exit 1; }
HIL_XML="$GS_ROOT/configs/hil_fastdds_x86.xml"
[[ -f $HIL_XML ]] || { echo "missing $HIL_XML"; exit 1; }
# USB NIC 地址变了就改 XML 里的 interfaceWhiteList,别改这里
if command -v ip >/dev/null 2>&1 && ! ip -4 addr show 2>/dev/null | grep -q "192.168.55.100"; then
  echo "WARN: 本机没有 192.168.55.100 的网卡 —— navcore USB 连着吗? hil_fastdds_x86.xml 白名单会失配" >&2
fi

bash "$GS_ROOT/kill_gsplat.sh" 2>/dev/null || true     # 别和 gz-parity face 共存
tmux kill-session -t "$SESSION" 2>/dev/null || true

cd "$WS_ROOT"
mkdir -p logs

if ! command -v ros2 >/dev/null 2>&1; then
  set +u; source /opt/ros/humble/setup.bash; set -u   # setup.bash 引用未绑定变量,挡 set -u
fi

win() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -d -t "$SESSION" -n "$1" -c "$WS_ROOT" "exec bash -i"
  else
    tmux new-session -d -s "$SESSION" -n "$1" -c "$WS_ROOT" "exec bash -i"
  fi
  tmux set-option -w -t "$SESSION:$1" remain-on-exit on
  tmux set-option -w -t "$SESSION:$1" automatic-rename off
  tmux send-keys -t "$SESSION:$1" "$2" Enter
}

# ---- sensor server:HIL 契约分辨率 544x640 / fx=302.779,infra1+color ----
# (color 非契约,但 build_map_node 的 4 路同步要 /camera/camera/color/*,
#  recorder 录 bag 时也会带上;两摄渲染 ~50ms 仍在 12Hz 预算内)
win gssim "export CUDA_HOME=$GS_CUDA_HOME PATH=\"$GS_CUDA_HOME/bin:\$PATH\" TORCH_CUDA_ARCH_LIST=12.0 GS_PLAYGROUND_ROOT=${GS_PLAYGROUND_ROOT:-/workspace/github/simulation/gs_playground}; $GS_VENV/bin/python -u $GS_ROOT/server/sensor_server.py --config $SCENE --rtf $RTF --cam-hz $CAM_HZ --cam-w 544 --cam-h 640 --cam-fy 302.779 --cams infra1,color ${SERVER_EXTRA[*]} 2>&1 | tee logs/hil_gssim.log"

RING=/dev/shm/gsplay_sensors.bin
echo "waiting for sensor ring $RING ..."
for _ in $(seq 1 180); do
  [[ -f $RING ]] && head -c 4 "$RING" | grep -q GSPG && break
  sleep 1
done
[[ -f $RING ]] || { echo "sensor ring never appeared -- check tmux: $SESSION:gssim"; exit 1; }

# ---- looper emulator:节点名 insight_full,直读 ring,发契约 ------------------
# DDS=CycloneDDS(实测定案):FastDDS 默认 65KB 数据报在 navcore 的
# NCM gadget 管道上重传风暴,在线录制只收 12-40% 帧;Cyclone 按片 MTU 级重传
# 满帧(图像/color 100%,depth 63-91%)。
# 两个坑(都实测过):①镜像烘焙的 CYCLONEDDS_URI 是 localhost-only,必须覆盖;
# ②Cyclone 配置里不能列 127.0.0.1 接口——它会选中 lo 当组播口并整体禁组播,
# 参与者对全网不可见(sensor:True 全是 pilot 缓存假象)。
# 不走 tmux 窗口:send-keys+交互 bash 的 env 落地不可靠,直接 nohup 起后台。
nohup env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 \
  CYCLONEDDS_URI=file://$GS_ROOT/configs/hil_cyclone_x86.xml \
  python3 -u $GS_ROOT/ros/looper_emu.py --fx 302.779 --report-every 15 \
  >> logs/hil_emu.log 2>&1 &
echo "emu pid $! (log: logs/hil_emu.log)"

echo
echo "HIL 起来了: tmux attach -t $SESSION   (gssim / emu 两个窗口)"
echo "验证:      navcore 侧 docker logs core-runtime 看 looper_bridge 三元组首中,"
echo "           curl http://192.168.55.1:8100/status 看 sensor:true"
echo "遛狗:      前端 http://192.168.55.1/ teleop,或 x86: ros2 topic pub /cmd_vel ..."
echo "真值:      bash gsplat/gs_state.sh   (ring GT)"
echo "收工:      navcore 侧 bash ~/navcore-deploy/pilot/hil_off.sh; 本机 tmux kill-session -t $SESSION"
