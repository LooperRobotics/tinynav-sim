#!/usr/bin/env bash
set -e

source /workspace/tinynav-sim/docker/shell-env.sh

workspace=/workspace/tinynav-sim
container_state="$workspace/.container"
build_base="$container_state/build"
install_base="$container_state/install"
cd "$workspace"

mkdir -p "$HOME" "$container_state/log"
printf '%s\n' 'source /workspace/tinynav-sim/docker/shell-env.sh' > "$HOME/.bashrc"
printf '%s\n' 'source "$HOME/.bashrc"' > "$HOME/.bash_profile"

build_core() {
  colcon build \
    --build-base "$build_base" \
    --install-base "$install_base" \
    --packages-select tinynav_cpp \
    --event-handlers console_direct+ \
    "$@"
}

require_models() {
  local model_dir=/tinynav/tinynav/models
  if ! compgen -G "$model_dir/*.plan" >/dev/null; then
    echo "No TensorRT .plan files found in $model_dir." >&2
    echo "Set TINYNAV_MODELS_DIR to a host directory containing the engines." >&2
    exit 2
  fi
}

run_core_tests() {
  local test_binary="$build_base/tinynav_cpp/tinynav_core_test"
  local test_workdir="$container_state/test-work"
  local fixture

  mkdir -p "$test_workdir/fixtures"
  shopt -s nullglob
  for fixture in "$workspace"/fixtures/*; do
    case "$(basename "$fixture")" in
      live_capture_smoke|live_capture_shape) continue ;;
    esac
    ln -sfn "$fixture" "$test_workdir/fixtures/$(basename "$fixture")"
  done
  shopt -u nullglob

  (cd "$test_workdir" && "$test_binary" "$@")
}

command_name=${1:-shell}
if (( $# > 0 )); then
  shift
fi

case "$command_name" in
  idle)
    exec sleep infinity
    ;;
  build)
    build_core "$@"
    ;;
  test)
    build_core
    run_core_tests "$@"
    ;;
  core-test)
    test_binary="$build_base/tinynav_cpp/tinynav_core_test"
    if [[ ! -x "$test_binary" ]]; then
      echo "Missing $test_binary; run the build command first." >&2
      exit 2
    fi
    run_core_tests "$@"
    ;;
  trt-shell)
    require_models
    exec bash "$@"
    ;;
  gazebo)
    export TINYNAV_INSTALL_PREFIX="$install_base"
    bash gazebo/run_simulator.sh "$@"
    exec tmux attach -t tinynav_sim
    ;;
  gazebo-detached)
    export TINYNAV_INSTALL_PREFIX="$install_base"
    exec bash gazebo/run_simulator.sh "$@"
    ;;
  gsplat)
    exec bash gsplat/run_gsplat.sh "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  exec)
    if (( $# == 0 )); then
      echo "Usage: tinynav-sim exec <command> [args...]" >&2
      exit 2
    fi
    exec "$@"
    ;;
  *)
    exec "$command_name" "$@"
    ;;
esac
