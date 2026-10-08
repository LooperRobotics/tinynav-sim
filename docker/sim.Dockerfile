# tinynav-sim: PURE SIMULATION image — gzsim (Ignition Fortress) + gsplat (3DGS).
# Contains nothing from the tinynav stack (no TRT, no /opt/venv, no GTSAM, no
# tinynav_cpp): the stack runs in its own container and talks DDS over
# loopback (single host) or the USB link (split-site).
#
# Build (repo root):
#   docker build -f docker/sim.Dockerfile -t tinynav-sim:x86_64 .
#
# Build-time network: aliyun pypi / pytorch-wheels / apt mirrors +
# pypi.motphys.com (the ONLY dependency aliyun does not mirror) +
# packages.osrfoundation.org + conda.anaconda.org. Do NOT route the build
# through a proxy — the domestic mirrors measure slower through one.
#
# Default GPU arch is RTX 5070 (Blackwell, sm_120). Other GPUs:
#   docker build --build-arg CUDA_ARCH="8.9" -f docker/sim.Dockerfile .
# (gsplat kernels are precompiled for CUDA_ARCH at build time; a mismatching
# runtime GPU transparently re-JITs via the baked nvcc, just slower.)
ARG CUDA_ARCH=12.0
FROM ros:humble-ros-base
ARG CUDA_ARCH
ENV DEBIAN_FRONTEND=noninteractive

# ---- apt: DEFAULT international sources (through the host's clash TUN) ------
# Measured on this rig: apt against the ALIYUN mirror crawls at ~300 KB/s while
# curl to the same host pulls 28 MB/s — an apt-client-specific throttle on the
# mirror side, not a routing problem (host-network build, IPv4-forced, both
# unchanged it). Against the default sources through clash, apt-get update
# takes ~10 s and packages flow at proxy speed. uv/pip below still use aliyun:
# uv's TLS stack is NOT throttled (35 MB/s measured, see the bootstrap probe).
# Package list mirrors the go2_sim image (the field-proven gzsim rig), minus
# the tinynav stack it used to carry alongside. jammy universe carries the
# full ignition-fortress set (libignition-gazebo6 6.18.0), no OSRF repo needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ignition-fortress \
      ros-humble-ros-gz \
      ros-humble-ros2-control ros-humble-gz-ros2-control ros-humble-ros2-controllers \
      ros-humble-joint-state-broadcaster ros-humble-joint-state-publisher \
      ros-humble-xacro ros-humble-robot-state-publisher \
      ros-humble-rviz2 \
      ros-humble-rmw-cyclonedds-cpp \
      python3-numpy python3-scipy python3-opencv python3-pil python3-pynput \
      libgl1-mesa-dri mesa-utils \
      python3-pip tmux \
 && rm -rf /var/lib/apt/lists/*

# ---- /opt/venv_gs: the gsplat stack (mirrors tools/probes/gsplat_humble_bootstrap.sh,
#      which is the container-verified install path). --system-site-packages keeps
#      the fused single-process option (rclpy + torch in one interpreter) open.
RUN pip3 install -i https://mirrors.aliyun.com/pypi/web/simple uv
RUN uv venv /opt/venv_gs --python /usr/bin/python3.10 --system-site-packages
RUN UV_HTTP_TIMEOUT=600 uv pip install --python /opt/venv_gs/bin/python \
      torch==2.7.0+cu128 \
      --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128 \
      --index-url https://mirrors.aliyun.com/pypi/web/simple
# unsafe-best-match: aliyun also carries motrixsim-core, but only the private
# index has the pinned dev build — uv's default first-match strategy refuses
# to look any further and fails the install.
RUN UV_HTTP_TIMEOUT=600 uv pip install --python /opt/venv_gs/bin/python \
      motrixsim-core==0.7.1.dev97295 \
      --index-url https://pypi.motphys.com/simple/ \
      --extra-index-url https://mirrors.aliyun.com/pypi/web/simple \
      --index-strategy unsafe-best-match
RUN UV_HTTP_TIMEOUT=600 uv pip install --python /opt/venv_gs/bin/python \
      gsplat==1.5.3 gaussian_renderer==0.2.0 numpy==2.2.6 scipy==1.15.3 \
      plyfile==1.1.3 onnxruntime==1.22.1 ninja pillow packaging \
      --index-url https://mirrors.aliyun.com/pypi/web/simple \
 && rm -rf /root/.cache/uv /root/.cache/pip

# ---- CUDA 12.8 nvcc via micromamba (the stock image CUDA is 12.2 and cannot
#      target sm_120) + the standard-layout shim gsplat's JIT expects.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba \
 && MAMBA_ROOT_PREFIX=/tmp/mamba /tmp/bin/micromamba create -y -p /opt/cuda-12.8 \
      -c https://conda.anaconda.org/nvidia -c conda-forge \
      cuda-version=12.8 cuda-nvcc cuda-cudart-dev cuda-cccl \
 && mkdir -p /opt/cuda-shim-gs && cd /opt/cuda-shim-gs \
 && ln -s ../cuda-12.8/bin bin \
 && ln -s ../cuda-12.8/targets/x86_64-linux/include include \
 && ln -s ../cuda-12.8/lib lib64 \
 && ln -s ../cuda-12.8/lib lib \
 && rm -rf /tmp/bin /tmp/mamba /root/.cache

# ---- gsplat kernel precompile (JIT would otherwise cost ~75 s on first run;
#      keyed under TORCH_EXTENSIONS_DIR so runtime users load it read-only).
#      ninja ships inside venv_gs — expose it system-wide so a runtime re-JIT
#      (GPU arch mismatch) also finds it, without putting the venv's python3
#      on the global PATH.
RUN ln -sf /opt/venv_gs/bin/ninja /usr/local/bin/ninja \
 && CUDA_HOME=/opt/cuda-shim-gs PATH=/opt/venv_gs/bin:/opt/cuda-shim-gs/bin:${PATH} \
      TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
      /opt/venv_gs/bin/python -c "from gsplat.cuda._backend import _C; print('gsplat kernel OK:', _C)" \
 && chmod -R 777 /opt/torch_extensions

# ---- DDS: bake the single-host loopback-unicast Cyclone config (verified
#      against VPN-TUN route hijacking; see docs/plan-sim-image-split.md).
#      Cross-host modes pass their own CYCLONEDDS_URI and override this.
COPY tools/probes/cyclone_localhost_unicast.xml /opt/dds/cyclone_localhost_unicast.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=file:///opt/dds/cyclone_localhost_unicast.xml \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    CUDA_HOME=/opt/cuda-shim-gs \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
# TORCH_CUDA_ARCH_LIST MUST match the precompile build above: torch keys the
# JIT cache on the arch flags, and with the var unset it recompiles for "all
# archs of visible cards" (77 s) instead of loading the baked kernel.
ENV PATH=/opt/cuda-shim-gs/bin:${PATH}

WORKDIR /workspace/tinynav-sim
CMD ["bash"]
