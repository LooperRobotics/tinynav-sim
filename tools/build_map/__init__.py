"""Route-B map builder: rosbag in, map-format-v2 + SigLIP2 semantic sidecar out.

Spec semantics come from reference/tinynav/core/build_map_node.py (AGENTS.md:
the reference snapshot is read-only and wins disagreements). Deviations:
- no v1 shelve/VideoDB stores; per-keyframe data lives in memory and is
  written through tools/mapio.MapWriter (v2 npy + semantic sidecar)
- semantic embedder is the SigLIP2 engine (L2 baked into the plan), not SigLIP1
- no TF/marker/rviz publishing (offline tool)
"""
