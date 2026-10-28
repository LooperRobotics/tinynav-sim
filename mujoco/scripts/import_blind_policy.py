#!/usr/bin/env python3
"""Import a go2-blind blind policy export as a tinynav-sim policy file.

go2-blind ships each robot as deploy/robots/<robot>/config/policy/velocity/
blind_v0/{exported/policy.onnx, params/deploy.yaml}. Its ONNX metadata names
the joints in model order, but the wheel-legged policies observe joint_vel
and output actions in policy order -- the 12 leg joints, then the 4 wheels --
and the metadata carries no action layout or wheel brake. This rewrites the
metadata (the graph is untouched) from deploy.yaml so sim/contract_v2.py
reads the policy as trained:

  joint_names, default_joint_pos, joint_stiffness, joint_damping
      policy order (deploy.yaml actions joint_names; legged: unchanged)
  action_layout      joint_pos[position]*0.25:<legs>[;wheel_vel[velocity]*s:<wheels>]
  wheel_brake_rad_s  deploy.yaml brake_below_speed (wheel-legged)
  gait_period, gait_ref_speed, gait_foot_xy, gait_march
                     the speed clock of deploy.yaml's gait_phase (go2-blind's
                     legged finals of 2026-10-27: ref_speed, foot_xy) and its
                     gait_march flag; absent for the fixed-clock exports
  source_md5         md5 of the go2-blind ONNX it came from

    python import_blind_policy.py <blind_v0 dir> <out.onnx>

The directory holds exported/policy.onnx and params/deploy.yaml (go2-blind's
deploy tree), or policy.onnx and deploy.yaml side by side (its hand-offs).
Needs onnx and pyyaml (not sim dependencies).
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import onnx
import yaml


def _csv(values) -> str:
    return ",".join(f"{float(v):.6g}" for v in values)


def import_policy(src: Path, out: Path) -> dict[str, str]:
    onnx_path = src / "exported" / "policy.onnx"
    deploy_path = src / "params" / "deploy.yaml"
    if not onnx_path.exists():
        onnx_path, deploy_path = src / "policy.onnx", src / "deploy.yaml"
    deploy = yaml.safe_load(deploy_path.read_text())
    model = onnx.load(onnx_path)
    meta = {p.key: p.value for p in model.metadata_props}
    model_names = meta["joint_names"].split(",")
    model_default = [float(v) for v in meta["default_joint_pos"].split(",")]

    actions = deploy["actions"]
    position = dict(actions["JointPositionAction"])
    velocity = actions.get("JointVelocityAction")
    # Legged exports name their joints by regex (".*"): model order.
    position["joint_names"] = [n for n in model_names if any(
        re.fullmatch(p, n) for p in position["joint_names"])] if any(
        not p.endswith("_joint") for p in position["joint_names"]) else list(position["joint_names"])
    if len(position["scale"]) == 1:
        position["scale"] = position["scale"] * len(position["joint_names"])
    order = list(position["joint_names"]) + (list(velocity["joint_names"]) if velocity else [])
    if sorted(order) != sorted(model_names):
        raise ValueError(f"{src}: deploy.yaml joints {order} vs ONNX {model_names}")
    default = [float(v) for v in deploy["default_joint_pos"]]
    for name, value in zip(order, default):
        if abs(model_default[model_names.index(name)] - value) > 1e-4:
            raise ValueError(f"{src}: default pose of {name} differs between deploy.yaml and ONNX")

    groups = []
    for term, kind, cfg in (("joint_pos", "position", position), ("wheel_vel", "velocity", velocity)):
        if cfg is None:
            continue
        scales = {float(s) for s in cfg["scale"]}
        if len(scales) != 1:
            raise ValueError(f"{src}: {term} needs one action scale, got {sorted(scales)}")
        groups.append(f"{term}[{kind}]*{scales.pop():g}:" + ",".join(cfg["joint_names"]))

    meta.update({
        "joint_names": ",".join(order),
        "default_joint_pos": _csv(default),
        "joint_stiffness": _csv(deploy["stiffness"]),
        "joint_damping": _csv(deploy["damping"]),
        "action_layout": ";".join(groups),
        "source_md5": hashlib.md5(onnx_path.read_bytes()).hexdigest(),
    })
    if velocity is not None:
        meta["wheel_brake_rad_s"] = f"{float(velocity['brake_below_speed']):.6g}"
    clock = (deploy.get("observations", {}).get("gait_phase") or {}).get("params") or {}
    if clock.get("ref_speed") is not None:
        # as written in deploy.yaml (B2's ref_speed 1.318333 has 7 digits)
        meta["gait_period"] = repr(float(clock["period"]))
        meta["gait_ref_speed"] = repr(float(clock["ref_speed"]))
        if clock.get("foot_xy"):
            meta["gait_foot_xy"] = ";".join(",".join(repr(float(v)) for v in foot)
                                            for foot in clock["foot_xy"])
        meta["gait_march"] = "1" if deploy.get("gait_march") else "0"
    del model.metadata_props[:]
    for key, value in meta.items():
        model.metadata_props.add(key=key, value=value)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="go2-blind .../policy/velocity/blind_v0 directory")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    meta = import_policy(args.src, args.out)
    print(f"{args.out}: {meta['action_layout'][:80]}... brake {meta.get('wheel_brake_rad_s')} "
          f"(from md5 {meta['source_md5'][:8]})")


if __name__ == "__main__":
    main()
