# viser web client builder: the vendored source under mujoco/viser_client
# (upstream v1.1.1 with the keyboard camera bindings removed at source
# level -- see that README) compiled by viser's own vite pipeline. Kept as
# a separate stage so the runtime image needs no node; the npmmirror
# registry default matches the apt/pypi mirror choices below.
FROM node:22-slim AS viser-client
ARG NPM_REGISTRY=https://registry.npmmirror.com
COPY mujoco/viser_client /viser_client
WORKDIR /viser_client
RUN npm ci --registry="$NPM_REGISTRY" --no-audit --no-fund && npm run build

# Overall-sim container: MuJoCo dog sim + wgpu splat + rclpy contract face.
#
# One image for the whole tinynav simulation side (hil.py publishes the
# looper contract straight into the shared host-network DDS domain via
# rclpy). Deliberately NOT from
# the uniflexai/tinynav base: this stack needs python>=3.11 for wgpu 0.32
# (humble = py3.10), and it needs none of torch/TensorRT/gz.
#
# Build (context = repo root; BuildKit required; the model/ asset bundle
# must be present — unzip model.zip at the repo root first: it carries the
# splat scene + go2 meshes, none of which are in git):
#   docker build -f docker/mujoco.Dockerfile -t mjsim:jazzy .
#
# Run: bash docker/run-mujoco-sim.sh [hil|sim|mj]   (wraps docker run with
# GPU / X11 / /dev/input / host-net flags)
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libegl1 libgles2 libvulkan1 \
      libx11-6 libxext6 libxcursor1 libxrandr2 libxinerama1 libxi6 libgl1 \
      ros-jazzy-rmw-cyclonedds-cpp \
      python3-pip python3-venv tini \
    && rm -rf /var/lib/apt/lists/*
# ros images ship no pip and ubuntu 24.04 marks the interpreter
# externally-managed (PEP 668) -- a container image is exactly the place
# where overriding that is the right call
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# python stack in a dedicated venv: the debian python carries rclpy (made
# visible via --system-site-packages) but its apt numpy blocks pip upgrades;
# the venv shadows it cleanly with the same numpy version pinned.
# uv for the heavy install: pip serially backtracks through large opencv
# wheels at mirror speed (~15 min here); uv resolves once and downloads in
# parallel. uv and the deps both come from the same aliyun index (PyPI
# mirror; ghcr.io is flaky on this network). gausscam is fetched from its
# exact official wheel URL instead of the index: aliyun lags on fresh
# releases and 0.1.5 is not synced yet. Swap this back to `gausscam==0.1.5`
# once the mirror carries it (the installed artifact is 0.1.5,
# sha256 240f88a20cac49e0a406f482f0aaf3ef39acce07881e0e48d7b6e36cfdb09efd).
# The pin matters: hil exercises the whole gausscam API at boot, so a bad
# version fails the image loudly rather than at run time.
RUN python3 -m pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ uv \
    && uv venv --system-site-packages -p /usr/bin/python3 /opt/mjsim \
    && uv pip install --python /opt/mjsim/bin/python \
      --index-url https://mirrors.aliyun.com/pypi/simple/ \
      numpy==1.26.4 \
      mujoco==3.14.0 onnxruntime opencv-python-headless wgpu==0.32.0 \
      viser==1.1.1 trimesh plyfile \
      https://files.pythonhosted.org/packages/76/95/b87b576f638b58d1a0104c27d7526cddf4f5fb0ec4a18743ea6f2f16c87c/gausscam-0.1.5-py3-none-any.whl \
    && rm -rf /root/.cache/uv

# viser's web client hardcodes keyboard camera controls (arrows rotate,
# KeyE elevates) with no upstream toggle (viser-project/viser#259 open) --
# arrows collide with the global /dev/input teleop, which fires regardless
# of browser focus. We ship the vendored mujoco/viser_client (upstream
# v1.1.1, bindings removed at source level in CameraControls.tsx) built by
# the node stage above. The sanity layer decodes the packed payload and
# refuses to build an image whose client still binds camera keys.
# WASD/KeyQ stay for camera work.
COPY --from=viser-client /viser_client/build/index.html \
     /opt/mjsim/lib/python3.12/site-packages/viser/client/build/index.html
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/* \
    && python3 - /opt/mjsim/lib/python3.12/site-packages/viser/client/build/index.html <<'PYEOF'
import re, subprocess, sys
raw = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r'data-p="([^"]*)"', raw, re.S)
assert m, "viser client is not in the expected packed layout"
ALPHA = "!#$%()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_abcdefghijklmnopqrstuvwxyz{|}~"
L = [0]*127
for i in range(88):
    L[ord(ALPHA[i])] = i
def de(s):
    n = len(s)//5; r = len(s) % 5
    out = bytearray(n*4 + (r-1 if r else 0)); o = 0
    for i in range(n):
        v = 0
        for c in s[i*5:(i+1)*5]:
            v = v*88 + L[ord(c)]
        out[o:o+4] = bytes((v >> sh & 255 for sh in (24, 16, 8, 0))); o += 4
    if r:
        v = 0
        for c in s[n*5:]:
            v = v*88 + L[ord(c)]
        out[o:] = bytes((v >> 8*j) & 255 for j in range(r-2, -1, -1))
    return bytes(out)
blob = subprocess.run(["zstd", "-d", "-c", "-"], input=de(m.group(1)),
                      capture_output=True, check=True).stdout
bound = (b"held.has(`Arrow" in blob or b"held.has(`KeyE`)" in blob
         or b"elevate(i,!1),r.has(`KeyE`)" in blob)
assert not bound, "viser client still binds camera keys -- fix mujoco/viser_client/src/CameraControls.tsx and rebuild the image"
print("[viser] client payload verified: camera keybindings absent")
PYEOF
ENV PATH=/opt/mjsim/bin:$PATH

COPY mujoco /workspace/tinynav-sim/mujoco
# DDS: bake the single-host loopback-unicast Cyclone config (same file as
# sim.Dockerfile; proven against VPN-TUN route hijacking -- without it
# large image samples degrade to ~4 Hz for peers on the host network).
COPY tools/probes/cyclone_localhost_unicast.xml /opt/dds/cyclone_localhost_unicast.xml
# model/ asset bundle (model.zip, not in git): splat scene + go2 meshes,
# baked AT THE REPO ROOT -- the same layout a host checkout has after
# unzipping model.zip, so the repo-root model/ lookups in plant/hil/view
# work unchanged in-container.
COPY model /workspace/tinynav-sim/model
COPY docker/mjsim-entrypoint.sh /usr/local/bin/mjsim
RUN chmod +x /usr/local/bin/mjsim

WORKDIR /workspace/tinynav-sim
# vsync kill: with __GL_SYNC_TO_VBLANK=0 the viewer's swap never blocks on
# an XWayland frame callback. NO GLVND/GLX vendor pins here: the viewer
# window renders via Mesa/iris on the Intel GPU (the X screen's GPU) and
# needs /dev/dri passed through at `docker run` (run-mujoco-sim.sh does);
# the GPU-heavy paths (wgpu splat, EGL depth) pick NVIDIA on their own.
# (Historical note: the "container viewer freezes after one frame" case was
# root-caused 2026-10-03 to hil.py never calling viewer.sync() -- NOT to
# GLX/NVIDIA. The no-pins guidance above still holds.)
ENV MUJOCO_GL=egl \
      __GL_SYNC_TO_VBLANK=0 \
      RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
      CYCLONEDDS_URI=file:///opt/dds/cyclone_localhost_unicast.xml \
      OPENBLAS_NUM_THREADS=1 \
      OMP_NUM_THREADS=1 \
      PYTHONPATH=/workspace/tinynav-sim/mujoco
# RMW pinned: jazzy defaults to Fast DDS whose discovery/logging threads
# burn extra cores beside the tinynav stack's CycloneDDS.
# CYCLONEDDS_URI pins DDS to loopback unicast (same file the sim.Dockerfile
# bakes): with defaults, large best-effort samples (the ~348 KB stereo
# frames) fragment across every interface including any VPN TUN and die
# mid-path -- subscribers saw 3.9 Hz of a clean 10 Hz while small topics
# stayed perfect. See tools/probes/cyclone_localhost_unicast.xml and
# docs/plan-sim-image-split.md.
# OPENBLAS/OMP pinned: numpy's bundled OpenBLAS spins its whole pool on
# every per-tick BLAS call (the PIE tick does tiny ones at 200 Hz) —
# unpinned the idle-ish sim burns 5+ cores on pool spin-wait (2026-10-04:
# 10.5ms-cpu/tick -> 0.9ms pinned).
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/mjsim"]
CMD ["hil"]
