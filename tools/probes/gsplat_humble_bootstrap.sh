#!/usr/bin/env bash
# Bootstrap pure-gsplat env inside a bare ros:humble-ros-base container.
# Mirrors gsplat/setup_env.sh's network-install path, but everything through
# Aliyun mirrors. Run INSIDE the probe container.
set -euxo pipefail

PY=/opt/venv_gs/bin/python
ALI_PY=https://mirrors.aliyun.com/pypi/web/simple

echo "== [0/6] apt + uv =="
if ! timeout 90 apt-get update; then
  sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g; s|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list
  apt-get update
fi
apt-get install -y --no-install-recommends python3-pip curl ca-certificates
pip3 install -i "$ALI_PY" uv
uv --version

echo "== [1/6] venv (--system-site-packages: exposes apt python3-rclpy for the fused test) =="
uv venv /opt/venv_gs --python /usr/bin/python3.10 --system-site-packages

echo "== [2/6] torch 2.7.0+cu128 (aliyun pytorch-wheels) =="
uv pip install --python "$PY" torch==2.7.0+cu128 \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128 \
  --index-url "$ALI_PY"

echo "== [3/6] motrixsim + gsplat stack =="
# unsafe-best-match: aliyun (public PyPI mirror) also carries motrixsim-core,
# but only the private index has the pinned dev build — uv's default
# first-match strategy then refuses to look any further.
uv pip install --python "$PY" motrixsim-core==0.7.1.dev97295 \
  --index-url https://pypi.motphys.com/simple/ --extra-index-url "$ALI_PY" \
  --index-strategy unsafe-best-match
uv pip install --python "$PY" \
  gsplat==1.5.3 gaussian_renderer==0.2.0 numpy==2.2.6 scipy==1.15.3 \
  plyfile==1.1.3 onnxruntime==1.22.1 ninja pillow packaging \
  --index-url "$ALI_PY"

echo "== [4/6] cuda 12.8 shim (nvcc copied in from host beforehand) =="
mkdir -p /opt/cuda-shim-gs && cd /opt/cuda-shim-gs
[[ -e bin ]] || ln -s ../cuda-12.8/bin bin
[[ -e include ]] || ln -s ../cuda-12.8/targets/x86_64-linux/include include
[[ -e lib64 ]] || ln -s ../cuda-12.8/lib lib64
[[ -e lib ]] || ln -s ../cuda-12.8/lib lib
/opt/cuda-12.8/bin/nvcc --version | tail -1

echo "== [5/6] gsplat JIT precompile + torch CUDA sanity =="
export CUDA_HOME=/opt/cuda-shim-gs PATH=/opt/cuda-shim-gs/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
"$PY" -c "from gsplat.cuda._backend import _C; print('gsplat kernel OK:', _C)"
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== [6/6] render self-check (church 3 frames, no ring) =="
cd /workspace/dm/tinynav-sim
GS_PLAYGROUND_ROOT=/workspace/github/simulation/gs_playground \
  "$PY" gsplat/server/sensor_server.py --out /tmp/gsdump --duration 4 --rtf 0 --no-ring
ls -la /tmp/gsdump
echo "BOOTSTRAP_OK"
