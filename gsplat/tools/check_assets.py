#!/usr/bin/env python3
"""Verify the gs_playground assets a gsplat scene needs, and print ship commands.

Everything the simulator loads is read from an EXTERNAL gs_playground checkout
(`GS_PLAYGROUND_ROOT`) — the gaussian plys, MJCF scenes and the locomotion
policy are far too big for git. This tool resolves the paths the way
sensor_server.py does, reports what is missing, and prints the exact tar/rsync
lines to hand the same set to a colleague.

Usage:
    python3 gsplat/tools/check_assets.py                 # every scene in gsplat/configs
    python3 gsplat/tools/check_assets.py --scene church
    python3 gsplat/tools/check_assets.py --gspg-root /path/to/gs_playground

Exit status: 0 = everything present, 1 = something missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GS_DIR = HERE.parent
WS_ROOT = GS_DIR.parent
CONFIG_DIR = GS_DIR / "configs"


def gspg_root() -> Path:
    env = os.environ.get("GS_PLAYGROUND_ROOT")
    if env:
        return Path(env)
    for cand in ("/workspace/github/simulation/gs_playground",
                 "/home/dm/workspace/github/simulation/gs_playground"):
        if Path(cand, "demo", "navigation").is_dir():
            return Path(cand)
    return Path("/workspace/github/simulation/gs_playground")


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n/1024**3:.2f}G"
    if n >= 1024 ** 2:
        return f"{n/1024**2:.1f}M"
    if n >= 1024:
        return f"{n/1024:.0f}K"
    return f"{n}B"


def check(scene_cfg: dict, nav: Path) -> list[tuple[str, Path, str]]:
    """Returns (label, resolved path, requirement note). 'required' vs 'optional'."""
    items: list[tuple[str, Path, str]] = []
    # A scene is shipped as its WHOLE directory: the MJCF pulls in siblings via
    # <include file="..."/> and mesh dirs via relative paths (church_collision.xml,
    # nav_scene_1/meshes), and the plys live under the same tree.
    scene_dir = (nav / scene_cfg["scene"]).parent.parent
    items.append((f"scene dir ({scene_cfg['scene'].split('/')[0]})", scene_dir, "required"))
    items.append(("go2 MJCF (mesh dir must sit beside it)",
                  nav / "models/robots/navigation/go2/go2_mjx.xml", "required"))
    items.append(("go2 meshes",
                  nav / "models/robots/navigation/go2/assets", "required"))
    items.append(("sensor rig (generate: gsplat/robots/go2/make_sensor_rig.py)",
                  nav / "models/robots/navigation/go2/go2_sensor_rig.xml", "generated"))
    items.append(("locomotion policy",
                  nav / "policies/go2_policy.onnx", "required"))
    for name, rel in (scene_cfg.get("scene_gaussians") or {}).items():
        items.append((f"scene gaussians [{name}]", nav / str(rel), "required"))
    robot_gs = str(scene_cfg.get("robot_gs_dir", "assets/go2"))
    items.append(("robot gaussians (per-link plys)", nav / robot_gs, "required"))
    items.append(("ALSA null config (audio-init noise)",
                  nav / "configs/asound-null.conf", "optional"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="all", help="church | nav1 | all | a config file name")
    ap.add_argument("--gspg-root", default=None)
    args = ap.parse_args()

    root = Path(args.gspg_root) if args.gspg_root else gspg_root()
    nav = root / "demo" / "navigation"
    print(f"GS_PLAYGROUND_ROOT = {root}")
    print(f"  (config paths resolve relative to {nav})\n")
    if not nav.is_dir():
        print(f"ERROR: {nav} not found — set GS_PLAYGROUND_ROOT to the gs_playground "
              f"checkout (see gsplat/README.md §资产依赖与路径配置)")
        return 1

    if args.scene == "all":
        cfgs = sorted(CONFIG_DIR.glob("*.json"))
    else:
        p = Path(args.scene)
        cfgs = [p if p.suffix == ".json" else CONFIG_DIR / f"{p}_go2.json"]
    if not cfgs:
        print(f"no configs found under {CONFIG_DIR}")
        return 1

    missing_total = 0
    ship_paths: set[str] = set()
    for cfg_path in cfgs:
        cfg = json.loads(cfg_path.read_text())
        print(f"== {cfg_path.name} (scene: {cfg.get('scene')}) ==")
        missing = 0
        for label, path, kind in check(cfg, nav):
            rel = path.relative_to(nav) if nav in path.parents or path == nav else path
            if path.exists():
                mark = "OK     "
                size = fmt_size(dir_size(path))
            elif kind == "generated":
                mark = "GEN    "   # not shipped; regenerated locally in one command
                size = "-"
            elif kind == "optional":
                mark = "opt-mis"
                size = "-"
            else:
                mark = "MISSING"
                size = "-"
                missing += 1
            print(f"  [{mark}] {str(rel):<58} {size}")
            if path.exists():
                ship_paths.add(str(rel))
        if missing:
            print(f"  -> {missing} missing item(s)")
            missing_total += missing
        print()

    if ship_paths:
        # only the paths that actually exist and are needed get shipped, and a
        # directory swallows its own children (church_scene covers its plys)
        cand = sorted(p for p in ship_paths
                      if not p.endswith("go2_sensor_rig.xml"))
        need = [p for p in cand
                if not any(p != q and p.startswith(q.rstrip("/") + "/") for q in cand)]
        total = sum(dir_size(nav / p) for p in need)
        print("== ship to a colleague ==")
        print(f"  # one tarball, {fmt_size(total)} uncompressed "
              f"({len(need)} paths):")
        print("  tar -C <this checkout>/demo/navigation -czf gsplat_assets.tgz \\")
        for p in need:
            print(f"      {p} \\")
        print()
        print(f"  # expected unpack size: {fmt_size(total)} (xzf)")
        print()
        print("  # receiver: unpack into their own gs_playground checkout, then")
        print("  tar -C <their checkout>/demo/navigation -xzf gsplat_assets.tgz")
        print("  python3 gsplat/robots/go2/make_sensor_rig.py   # regenerate the rig")
        print("  python3 gsplat/tools/check_assets.py           # re-verify")

    print(f"\nresult: {'ALL PRESENT' if missing_total == 0 else f'{missing_total} MISSING'}")
    return 0 if missing_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
