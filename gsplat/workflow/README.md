# MetaCam office workflow

This prototype builds an office scene from a MetaBag V2 capture and starts
TinyNav using the generated assets. The target is x64 Linux with an NVIDIA
GPU. The office calibration comes from `info/calibration.json` in the capture.

Build TinyNav through Pilot's native build first, including its Flutter web
app and ROS interfaces. Provide a Lightning-LM checkout and the Spirula
release executable. The workflow builds Lightning-LM and the workflow image
locally.

```bash
./build.sh /path/to/lightning /path/to/tinynav-pilot /path/to/spirula
METACAM_WEB_HOST=HOST_IP ./run.sh /path/to/capture /path/to/new-output /path/to/spirula-models
```

The image contains the workflow and app. The existing MQTT and WebRTC services
must be reachable from the host network. The runner prints the App URL after
the web endpoint responds and rendered frames advance. It leaves the container
running for interactive use. Use `docker stop tinynav-metacam-office` to stop it.

The capture and weights are mounted read-only. Logs, checkpoints and the
scene persist in the output directory. The runner records the current stage in `workflow-stage` and defaults to a
fresh output directory. For an interrupted left-eye SAM stage with completed
COLMAP alignment and both frame directories present, stop the old workflow
container and rerun with `METACAM_RESUME_MASKS=1` to restart masking.

On headless NVIDIA hosts where the GLX Vulkan driver fails to initialize,
create a copy of the host NVIDIA ICD manifest that uses
`libEGL_nvidia.so.0` as its `library_path`. Pass its absolute path as
`METACAM_VULKAN_ICD` when invoking the runner. The host manifest stays intact.
Check the SAM log for the NVIDIA device before relying on GPU throughput.
