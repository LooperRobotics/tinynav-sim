import json
from pathlib import Path

scene = Path("/scene")
training = scene / "outputs/scene-ba-30k"
model = training / "step-000030000.ckpt/splat.ply"
transform = json.loads((training / "scene_transform.json").read_text())
if transform["world_from_train"]["scale"] != 1:
    raise ValueError("Training scale differs from the collision geometry scale")
if model.stat().st_size < 1_000_000:
    raise ValueError(f"Unexpectedly small final splat: {model}")

assets = {
    "variant": "scene-ba-30k",
    "splat": str(model),
    "collision": str(scene / "geometry/collision-voxels.npz"),
    "mesh": str(scene / "geometry/lidar-surfaces.ply"),
}
for name in ("splat", "collision", "mesh"):
    if Path(assets[name]).stat().st_size == 0:
        raise ValueError(f"Empty asset: {assets[name]}")
output = scene / "sim-runtime/scene-assets.json"
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(assets, indent=2) + "\n")
temporary.replace(output)
