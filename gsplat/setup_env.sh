#!/usr/bin/env bash
# One-time bootstrap of the gsplat simulator environment INSIDE the rig
# container (docker exec tinynav bash, then run this). Everything lands under
# /opt and can be rolled back with:
#   rm -rf /opt/venv_gs /opt/cuda-12.8 /opt/cuda-shim-gs /root/.cache/torch_extensions
#
# What it creates (verified; see docs/gsplat-sim-progress.md):
#   /opt/venv_gs       python 3.10 venv: torch 2.7.0+cu128, gsplat 1.5.3,
#                      motrixsim-core (private pypi.motphys.com index),
#                      gaussian_renderer, onnxruntime (RL policy), numpy 2.2.6
#                      -- fully isolated from the ROS stack's /opt/venv
#   /opt/cuda-12.8     nvcc toolchain: the container's stock CUDA is 12.2 and
#                      cannot target the RTX 5070 (Blackwell, sm_120); gsplat
#                      JIT-compiles its kernels so it needs >= 12.8
#   /opt/cuda-shim-gs  standard-layout shim over the micromamba prefix
#                      (bin/ include/ lib64/), exported as CUDA_HOME
#   + the precompiled gsplat CUDA kernels in /root/.cache/torch_extensions
#
# Usage from the HOST (paths below are host paths; the script docker-execs):
#   bash gsplat/setup_env.sh            # full bootstrap (idempotent, skips done steps)
set -euo pipefail

CONTAINER=${CONTAINER:-tinynav}
GSPG_HOST=${GSPG_HOST:-/home/dm/workspace/github/simulation/gs_playground}

echo "== [1/5] CUDA 12.8 nvcc =="
if docker exec "$CONTAINER" test -x /opt/cuda-12.8/bin/nvcc; then
  echo "already present"
else
  # Copy the proven host micromamba prefix (self-contained, 1.3 G). If the
  # host copy is missing, fall back to installing from conda inside the
  # container (see docs/gsplat-sim-progress.md for the exact command).
  if [[ -x $GSPG_HOST/../../.local/cuda-12.8/bin/nvcc ]]; then
    docker cp /home/dm/.local/cuda-12.8 "$CONTAINER":/opt/cuda-12.8
  else
    docker exec "$CONTAINER" bash -c '
      set -e
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba
      /tmp/bin/micromamba create -y -p /opt/cuda-12.8 -c https://conda.anaconda.org/nvidia -c conda-forge \
        cuda-version=12.8 cuda-nvcc cuda-cudart-dev cuda-cccl'
  fi
fi
docker exec "$CONTAINER" bash -c '
  set -e
  mkdir -p /opt/cuda-shim-gs && cd /opt/cuda-shim-gs
  [[ -e bin    ]] || ln -s ../cuda-12.8/bin bin
  [[ -e include ]] || ln -s ../cuda-12.8/targets/x86_64-linux/include include
  [[ -e lib64 ]] || ln -s ../cuda-12.8/lib lib64
  [[ -e lib    ]] || ln -s ../cuda-12.8/lib lib
  /opt/cuda-12.8/bin/nvcc --version | tail -1'

echo "== [2/5] /opt/venv_gs =="
if docker exec "$CONTAINER" test -x /opt/venv_gs/bin/python && \
   docker exec "$CONTAINER" test -d /opt/venv_gs/lib/python3.10/site-packages/torch; then
  echo "already present"
elif [[ -d $GSPG_HOST/.venv/lib/python3.10/site-packages ]]; then
  # PRIMARY PATH (what was actually used first): stream the proven host
  # venv's site-packages into a fresh container venv bound to the container's
  # system python 3.10 (same version/ABI). 7 GB, a few minutes. The in-container
  # network install works but download.pytorch.org crawled (~20 MB/min).
  docker exec "$CONTAINER" bash -c '
    set -e
    export PATH=/root/.local/bin:$PATH
    uv venv /opt/venv_gs --python /usr/bin/python3.10'
  tar -C "$GSPG_HOST/.venv/lib/python3.10/site-packages" -cf - . | \
    docker exec -i "$CONTAINER" tar -C /opt/venv_gs/lib/python3.10/site-packages -xf -
  # prebuilt gsplat kernels from the host (same torch/python); remove to force a JIT
  docker exec "$CONTAINER" mkdir -p /root/.cache/torch_extensions
  [[ -d $HOME/.cache/torch_extensions/py310_cu128 ]] && \
    docker cp "$HOME/.cache/torch_extensions/py310_cu128" "$CONTAINER":/root/.cache/torch_extensions/
else
  # FALLBACK: network install. Mirrors gs_playground/pyproject.toml pins, minus
  # the jupyter/media extras the headless server never imports. torch must come
  # from the cu128 index; motrixsim-core only exists on the vendor's private
  # index; UV_HTTP_TIMEOUT is generous because the nvidia wheels are multi-GB.
  docker exec "$CONTAINER" bash -c '
    set -e
    export PATH=/root/.local/bin:$PATH UV_HTTP_TIMEOUT=600
    uv venv /opt/venv_gs --python /usr/bin/python3.10
    uv pip install --python /opt/venv_gs/bin/python torch==2.7.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python /opt/venv_gs/bin/python motrixsim-core==0.7.1.dev97295 \
      --index-url https://pypi.motphys.com/simple/ --extra-index-url https://mirrors.aliyun.com/pypi/web/simple
    uv pip install --python /opt/venv_gs/bin/python \
      gsplat==1.5.3 gaussian_renderer==0.2.0 numpy==2.2.6 scipy==1.15.3 \
      plyfile==1.1.3 onnxruntime==1.22.1 ninja pillow packaging \
      --index-url https://mirrors.aliyun.com/pypi/web/simple'
fi

echo "== [3/5] gsplat CUDA kernel precompile (once, ~90 s) =="
docker exec "$CONTAINER" bash -c '
  export CUDA_HOME=/opt/cuda-shim-gs PATH=/opt/cuda-shim-gs/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
  /opt/venv_gs/bin/python -c "from gsplat.cuda._backend import _C; print(\"gsplat kernel:\", _C)"'

echo "== [4/5] torch CUDA sanity =="
docker exec "$CONTAINER" /opt/venv_gs/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "== [5/5] render self-check (3 church frames, no ring) =="
docker exec "$CONTAINER" bash -c '
  export CUDA_HOME=/opt/cuda-shim-gs PATH=/opt/cuda-shim-gs/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
    GS_PLAYGROUND_ROOT=/workspace/github/simulation/gs_playground
  cd /workspace/dm/tinynav-sim
  /opt/venv_gs/bin/python gsplat/server/sensor_server.py --out /tmp/gsdump --duration 4 --rtf 0 --no-ring \
    && ls -la /tmp/gsdump'

echo "bootstrap done."
