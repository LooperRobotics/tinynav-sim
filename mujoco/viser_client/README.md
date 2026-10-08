# viser_client/ — vendored viser web client (v1.1.1, Apache-2.0)

The viser web client (React + three.js + vite), vendored from upstream tag
`v1.1.1` with ONE local patch:

- `src/CameraControls.tsx`: removed the keyboard camera bindings — the four
  Arrow keys (camera rotate) and `KeyE` (camera elevate). They collided with
  the global `/dev/input` teleop, which drives the dog regardless of browser
  focus. viser has no upstream toggle for this (viser-project/viser#259 is
  open); see the removed lines in any upstream diff of this file.
- `src/App.tsx`: the HDRI import path is repointed from upstream
  `../../_assets/...` to `../_assets/...` so the whole tree stays inside this
  directory (`_assets/hdri/potsdamer_platz_1k.jpg` vendored alongside).

## Why vendored

The mjsim image serves THIS client instead of the one packed in the viser
wheel. The python-side `viser` dependency is pinned to the matching version
(`viser==1.1.1` in docker/mujoco.Dockerfile) — the client speaks the server's
websocket protocol, so the two must move together.

## Build

```bash
cd mujoco/viser_client
npm ci --registry=https://registry.npmmirror.com   # once
npm run build                                      # -> build/index.html
```

`docker/mujoco.Dockerfile` builds this directory in a node stage on every
image build (no host node needed) and COPYs `build/index.html` over the
wheel's client; a decode assert in the image refuses a client whose camera
keys are still bound. For quick iteration: `npm run dev` (vite dev server)
expects the sim's websocket — simplest loop is `npm run build` + restart the
sim container.

Upstream: https://github.com/viser-project/viser (Apache-2.0; no NOTICE file
upstream — this README is the attribution).
