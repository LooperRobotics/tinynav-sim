ARG PREP_IMAGE=ubuntu:24.04
ARG NAV_IMAGE=tinynav-metacam-lightning:dev
FROM ${PREP_IMAGE} AS spirula-libs
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir /spirula-libs && cp -L /lib/x86_64-linux-gnu/{ld-linux-x86-64.so.2,libc.so.6,libm.so.6,libdl.so.2,libpthread.so.0,librt.so.1,libstdc++.so.6,libgcc_s.so.1,libgomp.so.1} /spirula-libs/

FROM ${NAV_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=spirula-libs /spirula-libs/ /opt/spirula/lib/
COPY bundle/spirula /opt/spirula/bin/spirula
COPY bundle/app/ /workspace/
COPY bundle/tinynav/ /tinynav/
RUN mkdir -p /workspace/drivers && ln -sfn /tinynav/tool/looper_bridge_node.py /workspace/drivers/looper_bridge_node.py
COPY metacam/ /opt/metacam/
COPY runtime/ /opt/metacam/runtime/
COPY run-in-container.sh /opt/metacam/run
RUN python3 -m venv /opt/metacam-python
RUN --mount=type=cache,target=/root/.cache/pip env -u PYTHONPATH /opt/metacam-python/bin/python -m pip install --index-url https://pypi.org/simple pip==25.3
RUN --mount=type=cache,target=/root/.cache/pip env -u PYTHONPATH /opt/metacam-python/bin/python -m pip install --timeout 90 --retries 8 --resume-retries 10 --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
RUN --mount=type=cache,target=/root/.cache/pip env -u PYTHONPATH /opt/metacam-python/bin/python -m pip install --timeout 90 --retries 8 --resume-retries 10 --index-url https://pypi.org/simple \
      'numpy==1.26.4' 'scipy==1.15.3' 'rosbags==0.11.9' 'open3d==0.19.0' \
      'pycolmap==4.2.1' 'opencv-python-headless==4.11.0.86' 'plyfile==1.1.3' \
      'gsplat==1.5.3' ninja
WORKDIR /scene
ENTRYPOINT ["/opt/metacam/run"]
