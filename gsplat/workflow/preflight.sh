#!/usr/bin/env bash
set -euo pipefail

fail() {
    printf 'Preflight: %s\n' "$*" >&2
    exit 1
}

if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s INPUT_DIRECTORY CALIBRATION_FILE OUTPUT_DIRECTORY LOCAL_IMAGE\n' "$0"
    exit 0
fi
[[ $# == 4 ]] || fail "Run $0 --help for usage."

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    fail 'This workflow requires x64 Linux.'
command -v docker >/dev/null || fail 'Docker CLI is missing.'
docker info >/dev/null || fail 'Docker daemon is unavailable.'
command -v nvidia-smi >/dev/null || fail 'NVIDIA driver is missing.'
nvidia-smi -L || fail 'No usable NVIDIA GPU on the host.'

[[ -d $1 ]] || fail "Capture directory does not exist: $1"
input=$(cd "$1" && pwd -P)
[[ -s "$input/metadata.yaml" ]] || fail 'Capture metadata.yaml is missing or empty.'
shopt -s nullglob
segments=("$input"/data/data_*)
(( ${#segments[@]} > 0 )) || fail 'No capture segments found under data/.'
for segment in "${segments[@]}"; do
    [[ -f $segment && -r $segment ]] || fail "Cannot read capture segment: $segment"
    IFS= read -r -n 16 magic < "$segment" || fail "Cannot read capture header: $segment"
    [[ $magic == '#METABAG V2.0' ]] || fail "Unsupported capture format: $segment"
done
[[ -f $2 && -r $2 && -s $2 ]] || fail "Calibration file is missing or unreadable: $2"
output=$(realpath -m -- "$3")
case "$output/" in
    "$input/"*) fail 'Choose an output directory outside the capture directory.' ;;
esac
mkdir -p -- "$output"
[[ -w $output ]] || fail "Output directory is not writable: $output"
df -h -- "$output"

platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$4") ||
    fail "Local image is missing: $4"
[[ $platform == linux/amd64 ]] || fail "Expected linux/amd64 image, found $platform."
docker run --rm --gpus all --entrypoint nvidia-smi "$4" -L ||
    fail 'The container cannot access the NVIDIA GPU.'
printf 'Preflight passed for %s (%s capture segments).\n' "$input" "${#segments[@]}"
