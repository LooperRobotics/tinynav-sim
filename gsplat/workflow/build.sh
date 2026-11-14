#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 || ${1:-} == --help ]]; then
    printf 'Usage: %s LIGHTNING_CHECKOUT TINYNAV_PILOT_CHECKOUT SPIRULA_BINARY\n' "$0"
    [[ ${1:-} == --help ]]
    exit
fi
workflow=$(cd "$(dirname "$0")" && pwd -P)
lightning=$(realpath -- "$1")
pilot=$(realpath -- "$2")
spirula=$(realpath -- "$3")
image=${METACAM_IMAGE:-tinynav-metacam:dev}
if [[ ${METACAM_REBUILD_LIGHTNING:-0} == 1 ]] ||
    ! docker image inspect tinynav-metacam-lightning:dev >/dev/null 2>&1; then
    docker build --build-arg "BUILD_JOBS=${METACAM_BUILD_JOBS:-2}" \
        --file "$workflow/Dockerfile.lightning" --tag tinynav-metacam-lightning:dev "$lightning"
fi

context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
cp "$workflow"/Dockerfile "$workflow"/run-in-container.sh "$context/"
cp -r "$workflow"/metacam "$workflow"/runtime "$context/"
mkdir -p "$context/bundle/"{app,tinynav}
cp "$spirula" "$context/bundle/spirula"
for directory in deploy interfaces pilot plugins servers tinynav-app vendor; do
    cp -r "$pilot/$directory" "$context/bundle/app/"
done
for directory in scripts tinynav tool; do
    cp -r "$pilot/tinynav/$directory" "$context/bundle/tinynav/"
done
docker build --tag "$image" "$context"
