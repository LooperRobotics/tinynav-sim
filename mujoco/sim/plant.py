#!/usr/bin/env python3
"""Plant models: map3 stairwell + flat smoke terrain, driven by the
stair-descent policy.

Adapts the PIE robot configuration from parkour_mjlab's native sim2sim
(deploy/pie/sim2sim/go2_pie_sim2sim.py, train/orient-ablation) onto the
splatsense map3 scene, mirroring training exactly:

- robot = training go2.xml (base_link, imu gyro sensor, *_collision geoms)
- FULL_COLLISION contacts (feet condim 3 / friction 0.6 / solimp tuned,
  other collision geoms condim 1, visual geoms off)
- position actuators with PD in the actuator model (gainprm=Kp,
  biasprm=[0,-Kp,-Kd]) -- ctrl IS q_target, no software PD
- scene option overrides: dt 0.005 ImplicitFast iter 10 / ls 20 / ccd 500
- front_depth camera per pie_contract (60x106 optical-Z, fovy 56.52)
- map3 collision friction forced to 0.6 (training terrain value; MuJoCo
  combines friction with element-wise max, feet are 0.6)
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent          # .../mujoco/sim
ASSETS = _HERE.parent / "assets"                 # .../mujoco/assets

from . import contract as C  # noqa: E402


def _preprocess_scene(scene_path: Path, workdir: Path) -> Path:
    """MuJoCo's <include> needs a rooted document, but collision fragments
    ship as bare <geom> lists (the motrixsim parser tolerated that). Copy
    the scene into workdir and wrap every <include file=...> target the
    same way; scenes without includes pass through unchanged."""
    workdir.mkdir(parents=True, exist_ok=True)
    text = scene_path.read_text()
    for m in re.finditer(r'<include\s+file="([^"]+)"\s*/?>', text):
        coll = scene_path.parent / m.group(1)
        (workdir / m.group(1)).write_text(
            "<mujoco>\n" + coll.read_text() + "\n</mujoco>\n")
    fixed = workdir / scene_path.name
    fixed.write_text(text)
    return fixed


def load_boxes(xml_path: str):
    """[(name, pos[3], half[3])] for every box geom in the collision model."""
    from xml.etree import ElementTree
    text = Path(xml_path).read_text()
    root = ElementTree.fromstring(text) if text.lstrip().startswith("<mujoco") \
        else ElementTree.fromstring("<mujoco>\n" + text + "\n</mujoco>")
    boxes = []
    for geom in root.iter("geom"):
        if geom.get("type", "box") != "box":
            continue
        boxes.append((
            geom.get("name", "?"),
            np.array([float(v) for v in geom.get("pos").split()]),
            np.array([float(v) for v in geom.get("size").split()])))
    return boxes


CATCH_FLOOR_Z = -0.05   # scene.xml catch plane


def ground_top_at(boxes, x: float, y: float, below_z: float):
    """Highest walkable top (step/landing/floor plane) under (x, y, below_z)."""
    best = _ACTIVE.catch_z
    for _, p, s in boxes:
        if abs(x - p[0]) <= s[0] and abs(y - p[1]) <= s[1]:
            top = p[2] + s[2]
            if top <= below_z + 1e-9 and top > best:
                best = top
    return best


GO2_PIE_XML = ASSETS / "go2" / "go2_pie.xml"
SCENE_XML = ASSETS / "mjcf" / "scene.xml"
COLLISION_XML = ASSETS / "mjcf" / "map3_collision.xml"


def _go2_meshdir() -> Path:
    """go2 visual meshes ship in the model/ asset bundle (model.zip unpacked
    at the repo root), NOT in git (mujoco/README.md, 资产来源). Search order:
    $ROBOT_ASSETS, the in-repo model/ bundle, then the legacy robot-assets
    checkout. The mjsim image bakes the bundle to /opt/model."""
    repo = _HERE.parents[1]                         # .../tinynav-sim
    candidates: list[Path] = []
    if os.environ.get("ROBOT_ASSETS"):
        candidates.append(Path(os.environ["ROBOT_ASSETS"]))
    candidates += [repo / "model",
                   Path.home() / "workspace" / "dm" / "robot-assets"]
    for root in candidates:
        d = root / "go2" / "assets"
        if d.is_dir():
            return d
    raise FileNotFoundError(
        "go2 meshes not found — unzip model.zip at the repo root "
        "(creates model/go2/) or point ROBOT_ASSETS at the robot-assets "
        "checkout")

# view_live proven spawn on the F1 slab, facing the L0 stair base (-x world).
SPAWN_XYZ = (-6.0, 4.0, 12.173)
SPAWN_YAW_DEG = 180.0
SPAWN_DROP = 0.30       # base_link height above ground top at spawn
SETTLE_STEPS = 300      # 1.5 s at dt 0.005, ctrl = default pose

# Top of the R2b flight: landing_F4_east (top 11.802), facing -y down the
# treads (8 steps to top 10.621 at y 2.94) -- the stairs-down eval start.
DOWN_SPAWN = (-8.75, 5.70, 12.2)
DOWN_YAW_DEG = 270.0

# D435i-style splat rig (compose_scene values) rides base_link so the
# wgpu renderer poses off the same bodies in every profile.
RIG_Y = {"infra1": 0.0255, "infra2": -0.0255, "color": 0.0}
RIG_POS = [0.19, 0.0, 0.09]
RIG_QUAT_WXYZ = [0.5, 0.5, -0.5, -0.5]
RIG_FOVY = 82.85

TERRAIN_FRICTION = (0.6, 0.005, 0.0001)

DEPTH_CAMERA_NAME = "front_depth"


@dataclass(frozen=True)
class Scene:
    """One navigable world: mjcf (+ optional collision fragment), catch
    floor, splat ply (model/splat/), named spawns, and the policy the hil
    single-policy host loads. Register a new scene = one dict entry."""

    xml: Path
    collision: Path | None
    catch_z: float
    ply: str
    spawns: dict
    default_spawn: str
    hil_policy: str


SCENES: dict[str, Scene] = {
    # map3 stairwell: 123 exact collision boxes (Blender rebuild), F1 slab
    # tops at z=0, catch floor 5 cm below.
    "map3": Scene(
        xml=ASSETS / "mjcf" / "scene.xml",
        collision=ASSETS / "mjcf" / "map3_collision.xml",
        catch_z=CATCH_FLOOR_Z,
        ply="map3_scene.ply",
        spawns={
            "down": (DOWN_SPAWN, DOWN_YAW_DEG),
            "f4": (SPAWN_XYZ, SPAWN_YAW_DEG),
            "flat1": ((-7.4, 4.0, 12.2), 180.0),
        },
        default_spawn="down",
        hil_policy="policy.onnx",          # stairs expert model_18997
    ),
    # MetaCam map_2 indoor (spirula_ba_s2 step-50000, 2.98M gaussians).
    # Spawn = the gsplat-validated low-capture-band spot (map2_go2.json
    # initial_qpos, yaw -11.49); ground top -1.20 from the spawn-footprint
    # gaussian histogram -- see map2_scene.xml, recalibrate ground there
    # AND here together by render. Flat indoor: hil loads the flat-omni
    # expert, view keeps dual-expert auto.
    "map2": Scene(
        xml=ASSETS / "mjcf" / "map2_scene.xml",
        collision=None,
        catch_z=-2.6,
        ply="map2_scene.ply",
        spawns={"low": ((3.21, -5.54, -0.755), -11.49)},
        default_spawn="low",
        hil_policy="policy_24000_flat_omni.onnx",
    ),
}
_ACTIVE = SCENES["map3"]


def get_scene(name: str) -> Scene:
    try:
        return SCENES[name]
    except KeyError:
        raise SystemExit(
            f"unknown scene {name!r}; choose {sorted(SCENES)}") from None


def active_scene() -> Scene:
    return _ACTIVE


@dataclass(frozen=True)
class Bindings:
    """Indices into MjModel/MjData the runtime and observers need."""

    root_qpos_adr: int
    gyro_adr: int
    acc_adr: int
    joint_qpos_adr: np.ndarray
    joint_dof_adr: np.ndarray
    actuator_ids: np.ndarray
    depth_camera_id: int
    depth_camera_geom_id: int  # -1 when the camera shell is absent (flat)


def _configure_robot_spec(spec: mujoco.MjSpec) -> None:
    """Apply the same joints, contacts, and actuators as PIE sim2sim."""
    thigh_ranges = {
        "FL_thigh_joint": (-1.5708, 2.2),
        "FR_thigh_joint": (-1.5708, 2.2),
        "RL_thigh_joint": (-0.5236, 2.2),
        "RR_thigh_joint": (-0.5236, 2.2),
    }
    for joint_name, joint_range in thigh_ranges.items():
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"go2 model is missing joint {joint_name!r}.")
        joint.range = joint_range

    foot_names = {f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL")}
    for geom in spec.geoms:
        name = geom.name or ""
        if name.endswith("_collision"):
            geom.contype = 1
            geom.conaffinity = 1
            geom.condim = 1
            geom.priority = 0
            if name in foot_names:
                geom.condim = 3
                geom.priority = 1
                geom.friction[0] = 0.6
                geom.solimp[:3] = (0.9, 0.95, 0.023)
        else:
            geom.contype = 0
            geom.conaffinity = 0

    for index, joint_name in enumerate(C.ACTUATED_JOINT_NAMES):
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"go2 model is missing joint {joint_name!r}.")
        joint.armature = float(C.JOINT_ARMATURE[index])

        stiffness = float(C.JOINT_STIFFNESS[index])
        damping = float(C.JOINT_DAMPING[index])
        effort = float(C.JOINT_EFFORT_LIMIT[index])
        actuator = spec.add_actuator(name=joint_name, target=joint_name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = stiffness
        actuator.biasprm[1] = -stiffness
        actuator.biasprm[2] = -damping
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.forcerange = (-effort, effort)


def _load_robot_spec() -> mujoco.MjSpec:
    # go2_pie.xml carries a meshdir="assets" placeholder and .obj mesh names;
    # the real meshes are the shared decimated MSH set (robot-assets repo),
    # spliced in here (path + extension).
    xml = GO2_PIE_XML.read_text()
    xml = xml.replace('meshdir="assets"', f'meshdir="{_go2_meshdir()}"')
    xml = re.sub(r'file="([a-z_0-9]+)\.obj"', r'file="\1.msh"', xml)
    spec = mujoco.MjSpec.from_string(xml)
    _configure_robot_spec(spec)
    return spec


def _add_depth_camera_and_shell(base) -> None:
    base.add_camera(
        name=DEPTH_CAMERA_NAME,
        pos=C.DEPTH_CAMERA_POS,
        quat=C.DEPTH_CAMERA_QUAT,
        fovy=C.DEPTH_FOVY_DEG,
    )
    # Camera shell visual, same dims/pose as PIE sim2sim (contype 0).
    base.add_geom(
        name="front_depth_camera_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.332784, 0.0, 0.074446),
        quat=C.DEPTH_CAMERA_QUAT,
        size=(0.045, 0.018, 0.012),
        rgba=(0.0, 0.0, 0.0, 1.0),
        contype=0,
        conaffinity=0,
        density=0.0,
        group=2,
    )


def _apply_plant_options(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = C.PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 500


def _bindings(model: mujoco.MjModel, shell_id: int) -> Bindings:
    def id_of(obj_type, name):
        i = mujoco.mj_name2id(model, obj_type, name)
        if i < 0:
            raise ValueError(f"model is missing {obj_type.name} {name!r}")
        return int(i)

    joint_qpos_adr = np.empty(C.NUM_ACTIONS, dtype=np.int32)
    joint_dof_adr = np.empty(C.NUM_ACTIONS, dtype=np.int32)
    actuator_ids = np.empty(C.NUM_ACTIONS, dtype=np.int32)
    for index, name in enumerate(C.ACTUATED_JOINT_NAMES):
        jid = id_of(mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = id_of(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if int(model.actuator_trnid[aid, 0]) != jid:
            raise ValueError(f"actuator {name!r} drives the wrong joint")
        joint_qpos_adr[index] = model.jnt_qposadr[jid]
        joint_dof_adr[index] = model.jnt_dofadr[jid]
        actuator_ids[index] = aid

    gyro_id = id_of(mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if int(model.sensor_dim[gyro_id]) != 3:
        raise ValueError("imu_ang_vel sensor must be 3-dim")
    acc_id = id_of(mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_acc")
    if int(model.sensor_dim[acc_id]) != 3:
        raise ValueError("imu_lin_acc sensor must be 3-dim")
    return Bindings(
        root_qpos_adr=int(
            model.jnt_qposadr[id_of(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")]
        ),
        gyro_adr=int(model.sensor_adr[gyro_id]),
        acc_adr=int(model.sensor_adr[acc_id]),
        joint_qpos_adr=joint_qpos_adr,
        joint_dof_adr=joint_dof_adr,
        actuator_ids=actuator_ids,
        depth_camera_id=id_of(mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_CAMERA_NAME),
        depth_camera_geom_id=shell_id,
    )


def _validate(model: mujoco.MjModel, b: Bindings) -> None:
    expected = (7 + C.NUM_ACTIONS, 6 + C.NUM_ACTIONS, C.NUM_ACTIONS)
    if (model.nq, model.nv, model.nu) != expected:
        raise ValueError(f"nq/nv/nu={(model.nq, model.nv, model.nu)}, want {expected}")
    lo = np.column_stack((-C.JOINT_EFFORT_LIMIT, C.JOINT_EFFORT_LIMIT))
    checks = (
        np.allclose(model.actuator_gainprm[b.actuator_ids, 0], C.JOINT_STIFFNESS),
        np.allclose(model.actuator_biasprm[b.actuator_ids, 1], -C.JOINT_STIFFNESS),
        np.allclose(model.actuator_biasprm[b.actuator_ids, 2], -C.JOINT_DAMPING),
        np.allclose(model.actuator_forcerange[b.actuator_ids], lo),
        np.allclose(model.dof_armature[b.joint_dof_adr], C.JOINT_ARMATURE),
    )
    if not all(checks):
        raise ValueError("compiled actuator contract does not match PIE")
    if not np.isclose(float(model.cam_fovy[b.depth_camera_id]), C.DEPTH_FOVY_DEG,
                      atol=1e-10):
        raise ValueError("depth camera fovy drifted from contract")
    if float(model.opt.timestep) != C.PHYSICS_DT:
        raise ValueError("scene option timestep is not PIE's 0.005")
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError("scene option integrator is not ImplicitFast")


def _compile(scene_spec: mujoco.MjSpec, with_shell: bool):
    shell = None
    if with_shell:
        for geom in scene_spec.geoms:
            if (geom.name or "") == "front_depth_camera_visual":
                shell = geom
                break
        if shell is None:
            raise ValueError("camera shell geom missing after add")
    model = scene_spec.compile()
    # EGL/depth render headroom (viewer screenshots + 60x106 depth pass).
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    model.vis.map.znear = 0.001 / float(model.stat.extent)
    b = _bindings(
        model,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_depth_camera_visual")
        if with_shell else -1,
    )
    _validate(model, b)
    return model, b


def build_model(scene_name: str = "map3") -> tuple[mujoco.MjModel, Bindings]:
    """Scene <name> + training go2 + PIE profile -> (model, bindings).
    Sets the active scene for reset_to_spawn's z-snapping."""
    global _ACTIVE
    sc = get_scene(scene_name)
    _ACTIVE = sc
    scene = mujoco.MjSpec.from_file(
        str(_preprocess_scene(sc.xml, ASSETS / ".compose"))
    )
    # Training terrain friction on the whole map (boxes + catch floor).
    for geom in scene.geoms:
        geom.friction[:] = TERRAIN_FRICTION

    robot = _load_robot_spec()
    yaw = np.radians(SPAWN_YAW_DEG)
    frame = scene.worldbody.add_frame(
        pos=list(SPAWN_XYZ), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    )
    base = next(b for b in robot.worldbody.bodies if b.name == "base_link")
    frame.attach_body(base, "")
    base_in_scene = next(b for b in scene.worldbody.bodies if b.name == "base_link")

    _add_depth_camera_and_shell(base_in_scene)
    for name, y in RIG_Y.items():
        cam = base_in_scene.add_camera(
            name=name, pos=[RIG_POS[0], y, RIG_POS[2]], quat=RIG_QUAT_WXYZ
        )
        cam.fovy = RIG_FOVY
    _apply_plant_options(scene)

    model, b = _compile(scene, with_shell=True)

    # spawn sanity: depth camera = base +x pitched down 20deg, i.e. world
    # (-cos20, 0, -sin20) at spawn yaw 180.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cid = b.depth_camera_id
    fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    want = (-np.cos(np.radians(20)), 0.0, -np.sin(np.radians(20)))
    if abs(fwd[0] - want[0]) > 0.01 or abs(fwd[2] - want[2]) > 0.01:
        raise ValueError(f"front_depth forward {fwd}, want {want}")
    return model, b


def build_map3_model() -> tuple[mujoco.MjModel, Bindings]:
    """Back-compat alias: build_model("map3")."""
    return build_model("map3")


def build_flat_model() -> tuple[mujoco.MjModel, Bindings]:
    """Ground plane + training go2 (upstream sim2sim 'flat' analogue)."""
    scene = mujoco.MjSpec.from_string(
        "<mujoco model='pie_flat_smoke'>"
        "<worldbody>"
        "<geom name='ground' type='plane' pos='0 0 0' size='20 20 0.1'"
        " rgba='0.55 0.55 0.52 1' friction='0.6 0.005 0.0001'/>"
        "</worldbody></mujoco>"
    )
    scene.worldbody.add_light(
        name="sun", pos=(0.0, 0.0, 5.0), dir=(0.0, 0.0, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
    )
    robot = _load_robot_spec()
    frame = scene.worldbody.add_frame(pos=(0.0, 0.0, 0.0))
    base = next(b for b in robot.worldbody.bodies if b.name == "base_link")
    frame.attach_body(base, "")
    base_in_scene = next(b for b in scene.worldbody.bodies if b.name == "base_link")
    _add_depth_camera_and_shell(base_in_scene)
    _apply_plant_options(scene)
    return _compile(scene, with_shell=True)


def reset_to_spawn(model: mujoco.MjModel, data: mujoco.MjData, b: Bindings,
                   settle: bool = True, pos: tuple | None = None,
                   yaw_deg: float | None = None) -> None:
    """Drop the dog at the scene spawn (map3), origin+0.32 (flat), or an
    explicit pos/yaw override (e.g. the R2b stairs-down start on
    landing_F4_east at (-8.75, 5.7) facing -y = yaw 270)."""
    mujoco.mj_resetData(model, data)
    root = b.root_qpos_adr
    if pos is not None:
        sc = _ACTIVE
        if sc.collision is not None:
            boxes = load_boxes(str(sc.collision))
            z = ground_top_at(boxes, *pos[:2], pos[2]) + SPAWN_DROP
        else:
            # scenes without a collision fragment: the hint IS a validated
            # standing pose (e.g. map2's gsplat initial_qpos)
            z = pos[2]
        yaw = np.radians(yaw_deg if yaw_deg is not None else 0.0)
        data.qpos[root:root + 3] = (pos[0], pos[1], z)
    elif model.ngeom > 100:  # map3: 123 collision boxes + floor -> find slab top
        boxes = load_boxes(str(SCENES["map3"].collision))
        z = ground_top_at(boxes, *SPAWN_XYZ[:2], SPAWN_XYZ[2]) + SPAWN_DROP
        yaw = np.radians(SPAWN_YAW_DEG)
        data.qpos[root:root + 3] = (SPAWN_XYZ[0], SPAWN_XYZ[1], z)
    else:
        z, yaw = 0.32, 0.0
        data.qpos[root:root + 3] = (0.0, 0.0, z)
    if pos is None and yaw_deg is not None:
        yaw = np.radians(yaw_deg)
    data.qpos[root + 3:root + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    data.qpos[b.joint_qpos_adr] = C.DEFAULT_JOINT_POS
    data.qvel[:] = 0.0
    data.ctrl[b.actuator_ids] = C.DEFAULT_JOINT_POS
    if settle:
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise FloatingPointError("state not finite after spawn reset")


def main() -> int:
    for name, fn in (("flat", build_flat_model), ("map3", build_map3_model)):
        model, b = fn()
        print(f"[{name}] nq/nv/nu={model.nq}/{model.nv}/{model.nu} "
              f"dt={model.opt.timestep} integ={model.opt.integrator} "
              f"cam_fovy={model.cam_fovy[b.depth_camera_id]:.4f} "
              f"geoms={model.ngeom} timestep_ok=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
