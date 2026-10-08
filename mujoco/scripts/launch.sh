#!/bin/bash
# Unified tmux launcher for the MuJoCo (PIE + wgpu splat) dog sim.
#
# Usage:
#   bash mujoco/scripts/launch.sh [sim|mj|hil] [options]
#
# Forms:
#   sim    view.py          headless viser web viewer + splat, keyboard drive
#                           (browser http://127.0.0.1:8012/; --preview adds an
#                           MJPEG dog-eye page :8888) -- the default
#   mj     physics_view.py  native MuJoCo viewer + PIE only (no splat) --
#                           physics debug (needs X)
#   hil    hil.py           headless looper-contract face on DDS (rclpy node
#                           /insight_full, same domain as the tinynav stack).
#                           Needs a ROS python -- the operative host is the
#                           mjsim container (bash docker/run-mujoco-sim.sh hil);
#                           a plain sim venv is refused below.
#                           (--cam-hz/--vio-hz/--imu-every/--preview/--p-port
#                           forwarded)
#
# Options:
#   --device NAME   wgpu adapter substring   (default: auto from nvidia-smi)
#   --spawn POS     down | f4 | flat1        (default: down)
#   --venv DIR      python venv              (default: probed, see below)
#   sim: --view web|glfw  --web-port N  --preview [--p-host --p-port]
#        (--view web is the entrypoint default; glfw needs X)
#
# venv probe order: $MJSIM_VENV, $ROOT/.venv*, sibling splatsense venvs
#   (../splatsense/.venv-wgpu on the work station, ../splatsense/.venv312 on
#   the home laptop). splatsense itself is found the same way and put on
#   PYTHONPATH, so no pip install is required anywhere.
#
# tmux session name = form name (sim / mj / hil), windows demo + log.
# End a run: tmux kill-session -t <session>   (kills everything, no leftovers)
set -u

FORM="sim"
DEVICE="" SPAWN="down" VENV="" VIEW="web"
HIL_EXTRA="" SIM_EXTRA=""

while [ $# -gt 0 ]; do
  case "$1" in
    sim|mj|hil) FORM="$1" ;;
    --device) DEVICE="$2"; shift ;;
    --spawn) SPAWN="$2"; shift ;;
    --venv) VENV="$2"; shift ;;
    --view) VIEW="$2"; SIM_EXTRA="$SIM_EXTRA --view $2"; shift ;;
    --web-port) SIM_EXTRA="$SIM_EXTRA --web-port $2"; shift ;;
    --p-host|--p-port)
      SIM_EXTRA="$SIM_EXTRA $1 $2"; HIL_EXTRA="$HIL_EXTRA $1 $2"; shift ;;
    --cam-hz|--vio-hz|--imu-every)
      HIL_EXTRA="$HIL_EXTRA $1 $2"; shift ;;
    --preview) SIM_EXTRA="$SIM_EXTRA --preview"; HIL_EXTRA="$HIL_EXTRA --preview" ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
DM_DIR="$(cd "$ROOT/../.." && pwd)"              # .../dm (sibling splatsense lives here)
SPLATSENSE_ROOT="${SPLATSENSE_ROOT:-$DM_DIR/splatsense}"

case "$FORM" in
  sim) ENTRY="view.py";          SESSION="sim"   ;;
  mj)  ENTRY="physics_view.py";  SESSION="mj"    ;;
  hil) ENTRY="hil.py";           SESSION="hil"   ;;
esac
LOG="/tmp/mjsim_${ENTRY%.py}.log"

# venv probe
if [ -z "$VENV" ]; then
  for cand in ${MJSIM_VENV:+"$MJSIM_VENV"} "$ROOT"/.venv* "$SPLATSENSE_ROOT"/.venv-wgpu "$SPLATSENSE_ROOT"/.venv312 /opt/mjsim; do
    [ -n "$cand" ] && [ -x "$cand/bin/python" ] && VENV="$cand" && break
  done
fi
[ -n "$VENV" ] || { echo "no venv found (set --venv or MJSIM_VENV)" >&2; exit 1; }
PY="$VENV/bin/python"

# GPU adapter substring
if [ -z "$DEVICE" ]; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  case "$GPU" in
    *5070*) DEVICE="5070" ;;
    *2060*) DEVICE="2060" ;;
    *)      DEVICE="5070" ;;
  esac
fi

# hil publishes DDS through rclpy -- a plain sim venv carries no ROS python,
# so probe up front and point at the container path instead of a confusing
# ImportError after the splat pipeline is already up.
if [ "$FORM" = "hil" ] && ! "$PY" -c "import rclpy" >/dev/null 2>&1; then
  echo "[launch][hil] $PY has no rclpy -- hil needs a ROS python" >&2
  echo "  operative path: bash docker/run-mujoco-sim.sh hil" >&2
  exit 1
fi

# keyboard capture needs the input group *in this session*; sg re-injects it
# when the desktop session predates `usermod -aG input`. hil is headless.
if [ "$FORM" = "hil" ]; then
  WRAP="exec"
elif id -Gn 2>/dev/null | tr ' ' '\n' | grep -qx input; then
  WRAP="exec"
else
  WRAP="sg"
fi

# viewer health probe for the X-needing forms only (mj always; sim when
# --view glfw): a wedged X compositor (stale surfaces from kill -9'd GL
# processes) makes glfwCreateWindow block forever -- detect it up front so
# the operator knows the native window will degrade. The default sim form
# (viser web) needs no X at all and skips the probe.
if [ "$FORM" = "mj" ] || { [ "$FORM" = "sim" ] && [ "$VIEW" = "glfw" ]; }; then
  export DISPLAY="${DISPLAY:-:1}" MUJOCO_GL=egl
  PROBE_OUT="$(timeout 8 "$PY" "$SCRIPTS/probe_viewer.py" 2>/dev/null || true)"
  if echo "$PROBE_OUT" | grep -q VIEWER-MAPPED; then
    echo "[launch] X healthy -> native viewer window"
  else
    echo "[launch] GLFW probe failed -> the run continues without the native window"
  fi
fi

# clean up any previous instance (patterns live in this file only)
tmux kill-session -t "$SESSION" 2>/dev/null
pkill -TERM -f "$ENTRY[.]py" 2>/dev/null; sleep 2
pkill -KILL -f "$ENTRY[.]py" 2>/dev/null; sleep 1

RUN="/tmp/mjsim_run_${FORM}.sh"
ENVV="MUJOCO_GL=egl DISPLAY=${DISPLAY:-:1} PYTHONPATH=$ROOT:$SPLATSENSE_ROOT"
EXTRA=""
[ "$FORM" = "hil" ] && EXTRA="$HIL_EXTRA"
[ "$FORM" = "sim" ] && EXTRA="$SIM_EXTRA"
cat > "$RUN" <<RUNEOF
#!/bin/bash
cd "$ROOT"
if [ "$WRAP" = "sg" ]; then
  exec sg input -c "$ENVV $PY -u $ENTRY $DEVICE --spawn $SPAWN$EXTRA"
else
  exec env $ENVV $PY -u $ENTRY $DEVICE --spawn $SPAWN$EXTRA
fi
RUNEOF
chmod +x "$RUN"

tmux new-session -d -s "$SESSION" -n demo "bash $RUN 2>&1 | tee $LOG"
tmux new-window -t "$SESSION" -n log "tail -f $LOG"

echo "[launch] tmux session '$SESSION' ($FORM, device $DEVICE, spawn $SPAWN)"
echo "[launch] attach: tmux attach -t $SESSION    end: tmux kill-session -t $SESSION"
if [ "$FORM" = "sim" ]; then
  if [ "$VIEW" = "web" ]; then
    echo "[launch] scene viewer: http://127.0.0.1:8012/  (viser; Reset/Stop in the GUI)"
    echo "[launch] dog-eye stream: off -- rerun with --preview for http://127.0.0.1:8888/"
  else
    echo "[launch] native viewer window (glfw)"
  fi
fi
if [ "$FORM" = "hil" ]; then
  echo "[launch] HIL: contract face on DDS (rclpy node /insight_full)"
  echo "[launch] verify (DDS peer, e.g. tinynav container):"
  echo "  docker exec tinynav bash -c 'source /opt/ros/humble/setup.bash && ros2 topic hz /camera/camera/vio_100hz'"
fi
