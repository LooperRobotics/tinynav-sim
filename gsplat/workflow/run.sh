#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 || ${1:-} == --help ]]; then
    printf 'Usage: %s CAPTURE_DIRECTORY OUTPUT_DIRECTORY SPIRULA_WEIGHTS_DIRECTORY\n' "$0"
    printf 'Set METACAM_ACCEPT_SAM_LICENSE=1 to accept the SAM 2.1 Apache 2.0 license in Spirula.\n'
    [[ ${1:-} == --help ]]
    exit
fi

workflow=$(cd "$(dirname "$0")" && pwd -P)
capture=$(realpath -- "$1")
output=$(realpath -m -- "$2")
weights=$(realpath -- "$3")
image=${METACAM_IMAGE:-tinynav-metacam:dev}
port=${METACAM_WEB_PORT:-8087}
name=${METACAM_CONTAINER:-tinynav-metacam}
address=${METACAM_WEB_HOST:-localhost}
public_port=${METACAM_APP_PORT:-8088}
gateway="${name}-gateway"
if [[ $port == "$public_port" ]]; then
    printf 'The internal frontend and gateway ports must differ.\n' >&2
    exit 1
fi
for weight in sam2.1_hiera_base_plus_f16.ggml grounding-dino-base.safetensors bert-base-uncased-vocab.txt; do
    [[ -s "$weights/$weight" ]] || { printf 'Missing weight: %s\n' "$weights/$weight" >&2; exit 1; }
done
"$workflow/preflight.sh" "$capture" "$capture/info/calibration.json" "$output" "$image"
if [[ -e "$output/workflow-stage" && ${METACAM_RESUME_MASKS:-0} != 1 && ${METACAM_RESUME_RENDER:-0} != 1 ]]; then
    printf 'Choose a fresh output directory: %s\n' "$output" >&2
    exit 1
fi

vulkan_options=()
for icd in "${METACAM_VULKAN_ICD:-}" /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [[ -r "$icd" ]]; then
        vulkan_options=(--mount "type=bind,source=$icd,target=/etc/vulkan/icd.d/nvidia_icd.json,readonly"
                        --env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
                        --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json)
        break
    fi
done

docker run --detach --init --name "$name" --gpus all --network host --shm-size=16g \
    "${vulkan_options[@]}" --env "METACAM_RESUME_MASKS=${METACAM_RESUME_MASKS:-0}" \
    --env "METACAM_RESUME_RENDER=${METACAM_RESUME_RENDER:-0}" \
    --env "METACAM_ACCEPT_SAM_LICENSE=${METACAM_ACCEPT_SAM_LICENSE:-0}" \
    --env "METACAM_WEB_PORT=$port" --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    --mount "type=bind,source=$workflow/run-in-container.sh,target=/opt/metacam/run,readonly" \
    --mount "type=bind,source=$capture,target=/capture,readonly" \
    --mount "type=bind,source=$output,target=/scene" \
    --mount "type=bind,source=$weights,target=/weights,readonly" \
    --mount "type=bind,source=$weights,target=/root/.cache/spirula-studio/models,readonly" \
    "$image" >/dev/null

last_stage=
while [[ $(docker inspect --format '{{.State.Running}}' "$name") == true ]]; do
    stage=$(cat "$output/workflow-stage" 2>/dev/null || true)
    if [[ $stage != "$last_stage" ]]; then
        printf 'Stage: %s\n' "$stage"
        last_stage=$stage
    fi
    if [[ $stage == launch && -s "$output/sim-runtime/front.jpg" && -s "$output/sim-runtime/state.json" ]] &&
        curl -fsS --max-time 3 "http://127.0.0.1:$port/" >/dev/null; then
        before=$(cat "$output/sim-runtime/state.json")
        sleep 2
        if [[ $before != "$(cat "$output/sim-runtime/state.json")" ]]; then
            docker run -d --name "$gateway" --network host --restart unless-stopped \
                --mount "type=bind,source=$workflow/app-gateway.conf,target=/tmp/app-gateway.conf,readonly" \
                -e "METACAM_APP_PORT=$public_port" -e "METACAM_WEB_PORT=$port" \
                nginx:alpine sh -ec 'sed -e "s/listen 8088;/listen ${METACAM_APP_PORT};/" -e "s/127.0.0.1:8087;/127.0.0.1:${METACAM_WEB_PORT};/" /tmp/app-gateway.conf >/tmp/nginx.conf; exec nginx -c /tmp/nginx.conf -g "daemon off;"' >/dev/null
            origin="http://127.0.0.1:$public_port"
            curl -fsS --retry 8 --retry-connrefused --max-time 5 "$origin/" >/dev/null
            curl -fsS --retry 8 --max-time 5 "$origin/api/sensor/image-topics" >/dev/null
            printf 'TinyNav App: http://%s:%s/\n' "$address" "$public_port"
            exit 0
        fi
    fi
    sleep 5
done
docker logs --tail 30 "$name" >&2
printf 'Workflow stopped during %s. Logs: %s/logs\n' "${last_stage:-startup}" "$output" >&2
exit 1
