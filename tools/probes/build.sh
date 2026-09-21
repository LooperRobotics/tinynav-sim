#!/usr/bin/env bash
# Build the standalone TRT probes (tools/probes/*.cpp) against the colcon
# build tree. Run INSIDE the rig container from the repo root — needs
# build/tinynav_cpp/libtinynav_trt.so (colcon build first), CUDA and TensorRT
# headers, i.e. a GPU rig. Binaries land next to the sources; fixtures/ stays
# data-only (gitignored), the sources are git-tracked.
#
#   bash tools/probes/build.sh
#   ./tools/probes/probe_lg fixtures/probe/out_failing        # replay a dump
set -euo pipefail
cd "$(dirname "$0")"   # tools/probes

INC="-I ../../src/tinynav_cpp/include -I /usr/include/opencv4 -I /usr/local/cuda/include"
LIB="-L ../../build/tinynav_cpp -L /usr/local/cuda/lib64 \
     -ltinynav_trt -lnvinfer -lcudart -lopencv_core -lopencv_imgproc \
     -lopencv_imgcodecs -lpthread"

[[ -d ../../build/tinynav_cpp ]] || {
  echo "ERROR: build/tinynav_cpp missing — colcon build tinynav_cpp first" >&2
  exit 1
}

for p in probe_lg probe_lg_race probe_sp probe_sp_race; do
  g++ -std=c++17 -O2 $INC "$p.cpp" -o "$p" $LIB
  echo "built $p"
done
