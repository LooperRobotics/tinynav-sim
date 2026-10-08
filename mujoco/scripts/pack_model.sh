#!/bin/bash
# Repack the model/ asset bundle (out-of-git: splat scenes + go2 meshes)
# into model.zip. Regenerates MANIFEST.sha256 from the current files first,
# so the manifest always describes exactly what ships. VERSION.txt is
# hand-written provenance -- edit it when contents change; the script only
# warns if it is missing.
#
# Usage: bash mujoco/scripts/pack_model.sh [output.zip]
#        (default output: <repo-parent>/model.zip, keeps the ~900 MB
#         artifact out of the repo tree)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"        # repo root
MODEL="$ROOT/model"
OUT="${1:-$MODEL/../model.zip}"

[[ -d $MODEL/splat && -d $MODEL/go2 ]] || {
  echo "error: $MODEL lacks splat/ or go2/ -- nothing to pack" >&2
  exit 1
}
[[ -f $MODEL/VERSION.txt ]] || \
  echo "warn: $MODEL/VERSION.txt missing -- add provenance before shipping" >&2

cd "$MODEL"
sha256sum splat/* $(find go2 -type f | sort) > MANIFEST.sha256
rm -f "$OUT"
zip -q -r "$OUT" .
echo "packed: $OUT ($(du -h "$OUT" | cut -f1))"
echo "receiver check: unzip, then sha256sum -c MANIFEST.sha256"
