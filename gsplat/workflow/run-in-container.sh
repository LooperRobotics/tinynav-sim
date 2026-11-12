#!/usr/bin/env bash
set -eo pipefail

export METACAM_SCENE=/scene
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
export PATH="/opt/metacam-python/bin:$CUDA_HOME/bin:$PATH"
tools=/opt/metacam
python=/opt/metacam-python/bin/python
ros_python=/opt/venv/bin/python
logs=/scene/logs
mkdir -p "$logs" /scene/calib /scene/sim-runtime

step() {
    local name=$1
    shift
    printf '%s\n' "$name" > /scene/workflow-stage
    printf 'Running %s (logs/%s.log)\n' "$name" "$name"
    "$@" > "$logs/$name.log" 2>&1
}

spirula() {
    /opt/spirula/lib/ld-linux-x86-64.so.2 \
        --library-path "/opt/spirula/lib:/usr/local/nvidia/lib64" \
        /opt/spirula/bin/spirula "$@"
}

source /opt/ros/humble/setup.bash
source /root/slam_ws/install/setup.bash
if [[ -f /workspace/interfaces/install/setup.bash ]]; then
    source /workspace/interfaces/install/setup.bash
fi
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-64}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI
if [[ ${METACAM_RESUME_RENDER:-0} == 1 ]]; then
    [[ -s /scene/sim-runtime/scene-assets.json ]]
else
    if [[ ${METACAM_RESUME_MASKS:-0} == 1 ]]; then
        [[ $(cat /scene/workflow-stage) == sam-left && -s /scene/colmap/align_stats.json &&
           -d /scene/sam_frames/left && -d /scene/sam_frames/right ]]
    else
        if [[ -e /scene/workflow-stage ]]; then
            echo 'Output already contains a workflow run. Use a fresh output directory.' >&2
            exit 1
        fi
        cp /capture/info/calibration.json /scene/calib/calibration-original.json
        step convert "$python" "$tools/convert_native_metabag.py"
        step map ros2 run lightning run_slam_offline --input_bag /scene/ros2_bag \
            --config /root/slam_ws/src/lightning-lm/config/metacam.yaml
        step calibration "$python" "$tools/prepare.py"
        step frames "$python" "$tools/convert_to_nerfstudio.py" --workers 4
        step map-qa "$python" "$tools/map_qa.py"
        step floor "$python" "$tools/floor_qa.py"
        step geometry "$python" "$tools/build_geometry.py"
        step pairs "$python" "$tools/gen_pairs.py"
        for stage in init extract match triangulate ba align; do
            options=()
            case "$stage" in
                extract|match) options+=(--threads 4) ;;
                ba) options+=(--threads 4 --max-iters 200) ;;
            esac
            step "ba-$stage" "$python" "$tools/colmap_prior_ba.py" "$stage" "${options[@]}"
        done
        step split "$python" "$tools/prepare_dataset.py" split
    fi
    sam_license=()
    if [[ ${METACAM_ACCEPT_SAM_LICENSE:-0} == 1 ]]; then
        sam_license=(--accept-license sam2=yes)
    fi
    for side in left right; do
        step "sam-$side" spirula sam track \
            --model /weights/sam2.1_hiera_base_plus_f16.ggml --detector gdino-base \
            --text person --frames "/scene/sam_frames/$side" --out "/scene/sam_masks/$side" \
            --memory-frames 4 --detect-every 10 --img-size 1024 --remove-prompted "${sam_license[@]}"
    done
    step combine "$python" "$tools/prepare_dataset.py" combine
    step dataset "$python" "$tools/prepare_dataset.py" adapter
    step train spirula train 3dgs --data /scene/spirula_ba --output-dir-prefix /scene/outputs \
        --output-dir-name scene-ba-30k --num-iterations 30000 --disable-viewer 1 \
        --keep-viewer-alive 0 --scene-center none --orientation-method none --eval-mode interval \
        --eval-interval 8 --save-eval-images 1 --eval-lpips 0 --cap-max 2000000 \
        --cache-images disk --apply-loss-for-mask 0 --save-full-checkpoint 1
    step export "$python" "$tools/export_scene.py"
fi
step render-smoke "$python" "$tools/runtime/server.py" --smoke
step validate "$python" "$tools/runtime/validate.py"

export PYTHONPATH="/workspace/vendor:/workspace:/tinynav:${PYTHONPATH:-}"
export TINYNAV_SENSOR_MODE=looper ROBOT_TYPE=go2 TINYNAV_DB_PATH=/scene/db
export TAILNET_NAME=metacam-sim CHASSIS_NET_IFACE=lo CHASSIS_NET_WAIT_SEC=0
export TINYNAV_PLANNER=library TINYNAV_FEATURE_MATCHER=lightglue
export UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONUNBUFFERED=1
unset TINYNAV_DEPTH_NO_RETURN_RANGE
mkdir -p "$TINYNAV_DB_PATH"

printf '%s\n' launch > /scene/workflow-stage
cd /workspace
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
start() {
    local name=$1
    shift
    "$@" > "$logs/$name.log" 2>&1 &
    pids+=("$!")
}
start renderer "$python" "$tools/runtime/server.py"
start sensors "$ros_python" "$tools/runtime/looper_emu.py" \
    --ring /scene/sim-runtime/sensors.bin --cmd-file /scene/sim-runtime/cmd.txt --fx 272
start bridge "$ros_python" /tinynav/tool/looper_bridge_node.py
for action in nav turn snapshot reloop link_fit; do
    start "server-$action" "$ros_python" -m "servers.$action" "/pilot/$action"
done
start frontend "$ros_python" /workspace/deploy/serve_web.py \
    /workspace/tinynav-app/frontend/build/web "${METACAM_WEB_PORT:-80}"
start webrtc "$ros_python" -m deploy.webrtc_pub
start pilot "$ros_python" -m pilot.main
wait -n
echo 'A simulation process exited. See /scene/logs.' >&2
exit 1
