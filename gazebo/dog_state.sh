#!/usr/bin/env bash
# One-shot robot state check — makes the "confirm position & heading before
# every new operation" discipline a single command instead of hand-run ign
# commands and mental yaw math. Born from the reloc session: the
# dog sat at y=1.95, yaw 104° (off the mapped line), reloc failures were
# CORRECT behavior, and a round of debugging was burned before anyone looked.
#
# Run INSIDE the rig container (needs `ign`; ROS env is sourced only for
# --slam). With no sim running the ign call times out and exits 1.
#
# Usage:
#   gazebo/dog_state.sh                            # gz ground truth only
#   gazebo/dog_state.sh --world yard               # skip world auto-detect
#   gazebo/dog_state.sh --slam                     # + /slam/odometry_visual
#   gazebo/dog_state.sh --map-dir output/map_v2    # + IN/OUT trajectory verdict
#
# The verdict is informational (exit status only reports sim reachability):
# IN means inside the mapped bounding box + margin. OUT means reloc failures
# are EXPECTED — reset (/slam/reset) or restart the rig before diagnosing.
set -uo pipefail

WORLD=""
MAP_DIR=""
MARGIN="0.5"
WITH_SLAM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --world) WORLD="$2"; shift 2 ;;
    --map-dir) MAP_DIR="$2"; shift 2 ;;
    --margin) MARGIN="$2"; shift 2 ;;
    --slam) WITH_SLAM=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- world auto-detect ------------------------------------------------------
if [[ -z "$WORLD" ]]; then
  WORLD=$(timeout 10 ign topic -l 2>/dev/null \
    | grep -oP '^/world/\K[a-zA-Z_0-9]+(?=/dynamic_pose/info)' | head -1)
  if [[ -z "$WORLD" ]]; then
    echo "ERROR: no /world/*/dynamic_pose/info topic — sim not running?" >&2
    echo "start it: bash gazebo/run_simulator.sh ... (inside the rig container)" >&2
    exit 1
  fi
fi

# --- gz ground truth --------------------------------------------------------
# dynamic_pose/info streams ignition.msgs.Pose_V; take the exact model "go2".
RAW=$(timeout 15 ign topic -e -t "/world/$WORLD/dynamic_pose/info" -n 5 2>/dev/null)
if [[ -z "$RAW" ]]; then
  echo "ERROR: no messages on /world/$WORLD/dynamic_pose/info" >&2
  exit 1
fi
read -r GX GY GZ GYAW GOX GOY OZ OW <<<"$(python3 - "$RAW" <<'PY'
import math, re, sys
txt = sys.argv[1]
m = re.search(r'name: "go2".*?position\s*\{([^}]*)\}.*?orientation\s*\{([^}]*)\}', txt, re.S)
if not m:
    print("0 0 0 0 0 0 0 0"); sys.exit()
def kv(s, k):
    mm = re.search(rf'\b{k}:\s*([-0-9.eE+]+)', s)
    return float(mm.group(1)) if mm else 0.0
x, y, z = (kv(m.group(1), k) for k in "xyz")
ow, ox, oy, oz = (kv(m.group(2), k) for k in ("w", "x", "y", "z"))
yaw = math.atan2(math.sin(2 * math.atan2(oz, ow)), math.cos(2 * math.atan2(oz, ow)))
print(f"{x:.3f} {y:.3f} {z:.3f} {math.degrees(yaw):.1f} {ox:.4f} {oy:.4f} {oz:.4f} {ow:.4f}")
PY
)"
if [[ -z "$GX" || ( "$GX" == "0" && "$GY" == "0" && "$GZ" == "0" && "$OW" == "0" ) ]]; then
  echo "ERROR: model \"go2\" not found in dynamic_pose/info" >&2
  exit 1
fi
echo "== gz ground truth (world: $WORLD) =="
echo "go2  x=$GX y=$GY z=$GZ  yaw=${GYAW}deg  quat(wxyz)=$OW $GOX $GOY $OZ"

# --- SLAM-side odom (optional) ----------------------------------------------
if [[ $WITH_SLAM == 1 ]]; then
  # set -u must be off for the ROS setup scripts (they reference unbound
  # vars and, in a non-interactive shell, that exits the whole script).
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  ODOM=$(timeout 8 ros2 topic echo --once --field pose.pose \
    /slam/odometry_visual nav_msgs/msg/Odometry 2>/dev/null)
  read -r SX SY SZ SYAW <<<"$(python3 - "$ODOM" <<'PY'
import math, re, sys
txt = sys.argv[1] if len(sys.argv) > 1 else ""
if not txt.strip():
    sys.exit()
def sec(name):
    m = re.search(name + r':[ \t]*\n((?:[ \t]+\w+: [^\n]+\n)+)', txt)
    return m.group(1) if m else ""
pos, ori = sec(r"position"), sec(r"orientation")
def kv(s, k, d=0.0):
    mm = re.search(rf'\b{k}: ([-0-9.eE+]+)', s or "")
    return float(mm.group(1)) if mm else d
x, y, z = kv(pos, "x"), kv(pos, "y"), kv(pos, "z")
yaw = math.atan2(math.sin(2 * math.atan2(kv(ori, "z"), kv(ori, "w"))),
                 math.cos(2 * math.atan2(kv(ori, "z"), kv(ori, "w"))))
print(f"{x:.3f} {y:.3f} {z:.3f} {math.degrees(yaw):.1f}")
PY
)"
  if [[ -n "$SX" ]]; then
    echo "== slam odom (/slam/odometry_visual) =="
    echo "odom x=$SX y=$SY z=$SZ  yaw=${SYAW}deg"
    # NOTE: odom lives in the SLAM odom frame, gz truth in the world frame —
    # the two yaws differ by the rig's fixed mounting offset (observed -90°).
    # Compare each side against its own expectation, never against the other.
  else
    echo "== slam odom: no message (stack down or topic absent) =="
  fi
fi

# --- trajectory verdict (optional) -------------------------------------------
# Map v2: pose_matrices.npy (n,4,4); a python-format poses.npy dict is not
# readable here on purpose — export v2 first (tools/export_map_v2.py).
if [[ -n "$MAP_DIR" ]]; then
  PM="$MAP_DIR/pose_matrices.npy"
  [[ -f "$PM" ]] || PM="$MAP_DIR/poses.npy"
  if [[ -f "$PM" ]]; then
    read -r X0 X1 Y0 Y1 VERDICT DETAIL <<<"$(/opt/venv/bin/python3 - "$PM" "$GX" "$GY" "$MARGIN" <<'PY'
import sys
import numpy as np
pm, gx, gy, margin = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
p = np.load(pm)
if p.dtype == object:  # python-format pickled dict — flatten the values
    p = np.stack(list(np.load(pm, allow_pickle=True).item().values()))
x, y = p[:, 0, 3], p[:, 1, 3]
x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
inside = (x0 - margin <= gx <= x1 + margin) and (y0 - margin <= gy <= y1 + margin)
detail = "in box"
if not inside:
    ax, why = [], []
    if not (x0 - margin <= gx <= x1 + margin):
        why.append(f"x={gx} outside [{x0 - margin:.2f}, {x1 + margin:.2f}]")
    if not (y0 - margin <= gy <= y1 + margin):
        why.append(f"y={gy} outside [{y0 - margin:.2f}, {y1 + margin:.2f}]")
    detail = "; ".join(why)
print(f"{x0:.2f} {x1:.2f} {y0:.2f} {y1:.2f} {'IN' if inside else 'OUT'} {detail}")
PY
)"
    echo "== mapped trajectory ($PM) =="
    echo "traj bbox x=[$X0, $X1] y=[$Y0, $Y1]  margin=${MARGIN}m"
    echo "verdict: $VERDICT ($DETAIL)"
    if [[ "$VERDICT" == "OUT" ]]; then
      echo "NOTE: off the mapped trajectory — reloc failures here are EXPECTED behavior."
      echo "      recover with: ros2 topic pub --once /slam/reset std_msgs/msg/Empty (soft)"
      echo "      or restart the rig (run_simulator.sh re-spawns at origin, yaw +x)."
    fi
  else
    echo "== mapped trajectory: $PM not found (need a v2-exported map dir) =="
  fi
fi
