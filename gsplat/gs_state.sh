#!/usr/bin/env bash
# One-shot robot state check for the gsplat rig — the counterpart of
# gazebo/dog_state.sh, reading ground truth from the shared-memory ring
# (ring v2 GT channel) instead of `ign topic /world/*/dynamic_pose/info`.
# Same discipline: run BEFORE every new sim operation; an off-trajectory dog
# makes reloc failures EXPECTED behavior, not bugs.
#
# Run INSIDE the rig container. With no server running the ring is stale or
# absent — the script says so and exits 1.
#
# Usage:
#   gsplat/gs_state.sh                            # ground truth only
#   gsplat/gs_state.sh --slam                     # + /slam/odometry_visual
#   gsplat/gs_state.sh --map-dir output/map_v2    # + IN/OUT trajectory verdict
#   gsplat/gs_state.sh --yaw-deg -90              # rotate GT into the map frame
#                                                 # (church: VIO removes the +90
#                                                 # spawn heading; gz rigs align
#                                                 # by construction and need 0)
#   gsplat/gs_state.sh --ring /dev/shm/gsplay_sensors.bin
set -uo pipefail

RING="/dev/shm/gsplay_sensors.bin"
MAP_DIR=""
MARGIN="0.5"
YAW_DEG="0"
WITH_SLAM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ring) RING="$2"; shift 2 ;;
    --map-dir) MAP_DIR="$2"; shift 2 ;;
    --margin) MARGIN="$2"; shift 2 ;;
    --yaw-deg) YAW_DEG="$2"; shift 2 ;;
    --slam) WITH_SLAM=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- ground truth from the ring ----------------------------------------------
if [[ ! -f "$RING" ]]; then
  echo "ERROR: sensor ring $RING missing — gsplat server not running?" >&2
  echo "start it: bash gsplat/run_gsplat.sh ... (inside the rig container)" >&2
  exit 1
fi
read -r GX GY GZ GYAW GOX GOY OZ OW GAGE <<<"$(/opt/venv/bin/python3 - "$RING" <<'PY'
import math, struct, sys
import numpy as np
buf = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")
if bytes(buf[0:4]) != b"GSPG":
    print("0 0 0 0 0 0 0 0 0"); raise SystemExit
gt_seq = struct.unpack_from("<Q", buf, 56)[0]
gt_time = struct.unpack_from("<d", buf, 64)[0]
n_imu = struct.unpack_from("<I", buf, 48)[0]
gt_base = 128 + (544 * 480 * 5) * 4 + 48 * n_imu
v = struct.unpack_from("<d3d4d", buf, gt_base)
x, y, z = v[1:4]
qx, qy, qz, qw = v[4:8]
yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
import time
age = max(0.0, time.time() - gt_time)   # sim time epoch == process start, so
                                        # this is only a liveness hint, not wall age
print(f"{x:.3f} {y:.3f} {z:.3f} {math.degrees(yaw):.1f} "
      f"{qx:.4f} {qy:.4f} {qz:.4f} {qw:.4f} {gt_seq}")
PY
)"
if [[ -z "$GX" || ( "$GX" == "0" && "$GY" == "0" && "$OW" == "0" && "$GAGE" == "0" ) ]]; then
  echo "ERROR: ring has no ground-truth samples (server predates ring v2, or crashed at boot)" >&2
  exit 1
fi
echo "== gs ground truth (ring GT, seq=$GAGE) =="
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
                 math.cos(2 * math.atan2(kv(ori, "z"), kv(ori, "w")))) if kv(ori, "w") or kv(ori, "z") else 0.0
print(f"{x:.3f} {y:.3f} {z:.3f} {math.degrees(yaw):.1f}")
PY
)"
  if [[ -n "$SX" ]]; then
    echo "== slam odom (/slam/odometry_visual) =="
    echo "odom x=$SX y=$SY z=$SZ  yaw=${SYAW}deg"
    # NOTE: odom lives in the SLAM odom frame, gt above in the MJCF world
    # frame — compare each side against its own expectation, never against
    # the other (fixed mounting offset between the two, same as gz).
  else
    echo "== slam odom: no message (stack down or topic absent) =="
  fi
fi

# --- trajectory verdict (optional) -------------------------------------------
# Two independent checks:
#  * SLAM odom vs bbox: odom IS the map frame (same spawn-relative convention
#    as when the map was built) — the exact verdict, no rotation needed.
#    Requires --slam (the odom read) plus --map-dir.
#  * GT vs bbox with --yaw-deg: approximate (VIO removes its ESTIMATED spawn
#    heading, ~94 deg for a configured 90 — a few degrees of residual x ~0.7 m
#    at 10 m); useful when the stack is down.
bbox_verdict() {  # $1 pm-file $2 x $3 y $4 margin $5 rot-deg -> sets X0..DETAIL
  read -r X0 X1 Y0 Y1 VERDICT DETAIL <<<"$(/opt/venv/bin/python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import math, sys
import numpy as np
pm, gx, gy, margin, yaw_deg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
gx, gy = c * gx - s * gy, s * gx + c * gy
p = np.load(pm)
if p.dtype == object:  # python-format pickled dict — flatten the values
    p = np.stack(list(np.load(pm, allow_pickle=True).item().values()))
x, y = p[:, 0, 3], p[:, 1, 3]
x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
inside = (x0 - margin <= gx <= x1 + margin) and (y0 - margin <= gy <= y1 + margin)
detail = "in box"
if not inside:
    why = []
    if not (x0 - margin <= gx <= x1 + margin):
        why.append(f"x={gx:.2f} outside [{x0 - margin:.2f}, {x1 + margin:.2f}]")
    if not (y0 - margin <= gy <= y1 + margin):
        why.append(f"y={gy:.2f} outside [{y0 - margin:.2f}, {y1 + margin:.2f}]")
    detail = "; ".join(why)
print(f"{x0:.2f} {x1:.2f} {y0:.2f} {y1:.2f} {'IN' if inside else 'OUT'} {detail}")
PY
)"
}

if [[ -n "$MAP_DIR" ]]; then
  PM="$MAP_DIR/pose_matrices.npy"
  [[ -f "$PM" ]] || PM="$MAP_DIR/poses.npy"
  if [[ -f "$PM" ]]; then
    echo "== mapped trajectory ($PM) =="
    if [[ -n "$SX" ]]; then
      # exact: odom IS the map frame (spawn-relative, like the map build)
      bbox_verdict "$PM" "$SX" "$SY" "$MARGIN" 0
      echo "traj bbox x=[$X0, $X1] y=[$Y0, $Y1]  margin=${MARGIN}m"
      echo "verdict (slam odom, map frame — exact): $VERDICT ($DETAIL)"
      ODOM_VERDICT="$VERDICT"
    fi
    # approximate: GT rotated by --yaw-deg (residual = VIO's estimated spawn
    # heading vs configured; useful when the stack is down)
    bbox_verdict "$PM" "$GX" "$GY" "$MARGIN" "$YAW_DEG"
    echo "verdict (gt, --yaw-deg=${YAW_DEG} — approximate): $VERDICT ($DETAIL)"
    if [[ "${ODOM_VERDICT:-$VERDICT}" == "OUT" ]]; then
      echo "NOTE: off the mapped trajectory — reloc failures here are EXPECTED behavior."
      echo "      recover with: restart the rig (run_gsplat.sh re-spawns at the scene"
      echo "      config's initial_qpos; kill_gsplat.sh first if the ring is stale)."
    fi
  else
    echo "== mapped trajectory: $PM not found (need a v2 map dir) =="
  fi
fi
