#!/usr/bin/env python3
"""Plant models: map3 stairwell + flat smoke terrain, driven by the
stair-descent policy.

The robot is a registry entry (sim/robots.py). go2 keeps the PIE profile
below; the other robots load their training MJCF (actuators and collision
setup already in it) and only get the cameras and scene options added here.

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

import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent          # .../mujoco/sim
ASSETS = _HERE.parent / "assets"                 # .../mujoco/assets

from . import contract as C  # noqa: E402
from .robots import DEFAULT_ROBOT, Robot, get_robot  # noqa: E402


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


def ground_top_at(boxes, x: float, y: float, below_z: float,
                  catch_z: float | None = None):
    """Highest walkable top (step/landing/floor plane) under (x, y, below_z);
    catch_z: the scene's catch floor (the last built scene's by default)."""
    best = _ACTIVE.catch_z if catch_z is None else catch_z
    for _, p, s in boxes:
        if abs(x - p[0]) <= s[0] and abs(y - p[1]) <= s[1]:
            top = p[2] + s[2]
            if top <= below_z + 1e-9 and top > best:
                best = top
    return best


GO2_PIE_XML = ASSETS / "go2" / "go2_pie.xml"
SCENE_XML = ASSETS / "mjcf" / "scene.xml"
COLLISION_XML = ASSETS / "mjcf" / "map3_collision.xml"


def _meshdir(robot: Robot) -> Path:
    """Visual meshes ship in the model/ asset bundle (model.zip unpacked at
    the repo root), NOT in git (mujoco/README.md, 资产来源). Search order:
    $ROBOT_ASSETS, the in-repo model/ bundle, then the legacy robot-assets
    checkout. The mjsim image bakes the bundle to /opt/model."""
    d = robot.mesh_dir()
    if d is None:
        raise FileNotFoundError(
            f"{robot.name} meshes not found — unzip model.zip at the repo root "
            f"(creates model/{robot.name}/) or point ROBOT_ASSETS at the "
            "robot-assets checkout")
    return d

# view_live proven spawn on the F1 slab, facing the L0 stair base (-x world).
SPAWN_XYZ = (-6.0, 4.0, 12.173)
SPAWN_YAW_DEG = 180.0
SPAWN_DROP = 0.30       # base_link height above ground top at spawn
# 0.2 s at dt 0.005, ctrl = default pose: enough to land on the feet. The
# policies are trained dropping from the spawn height and acting at once; a
# longer hold at the default pose (1.5 s) sank the Go2-W (leg kp 20) to a
# 0.29 m crouch from its 0.40 m stance, a start it had never seen, and it fell.
SETTLE_STEPS = 40

# Top of the R2b flight: landing_F4_east (top 11.802), facing -y down the
# treads (8 steps to top 10.621 at y 2.94) -- the stairs-down eval start.
DOWN_SPAWN = (-8.75, 5.70, 12.2)
DOWN_YAW_DEG = 270.0

# D435i-style splat rig (compose_scene values) rides base_link so the
# wgpu renderer poses off the same bodies in every profile.
RIG_Y = {"infra1": 0.0255, "infra2": -0.0255, "color": 0.0}
RIG_POS = [0.19, 0.0, 0.09]
# Other heads: the rig keeps Go2's offset from the depth camera.
RIG_FROM_DEPTH_CAMERA = np.subtract(RIG_POS, C.DEPTH_CAMERA_POS)
# Camera shell visual (PIE sim2sim dims), same offset from the camera.
SHELL_FROM_DEPTH_CAMERA = np.subtract((0.332784, 0.0, 0.074446),
                                      C.DEPTH_CAMERA_POS)
RIG_QUAT_WXYZ = [0.5, 0.5, -0.5, -0.5]
RIG_FOVY = 82.85

# mesh-depth camera: infra1's pose at the LOOPER contract's vertical fovy,
# so a mujoco-rendered depth is pixel-aligned with the contract K. The rig
# cameras above carry the PIE rig's own fovy and feed the splat pipeline as
# POSES ONLY -- their intrinsics are never projected through.
CONTRACT_DEPTH_CAMERA = "infra1_contract"

TERRAIN_FRICTION = (0.6, 0.005, 0.0001)

DEPTH_CAMERA_NAME = "front_depth"
SHELL_GEOM_NAME = "front_depth_camera_visual"
SHELL_HALF_SIZE = (0.045, 0.018, 0.012)   # PIE sim2sim's camera shell


@dataclass(frozen=True)
class Scene:
    """One navigable world: mjcf (+ optional collision fragment), catch
    floor, splat ply (model/splat/) and named spawns. Policies belong to
    the robot (sim/robots.py), not the scene. Register a new scene = one
    dict entry (the registry's order is the web panel's map dropdown's).
    xml is None only for the procedural "flat" smoke terrain (built from
    a string, no mjcf file)."""

    xml: Path | None
    collision: Path | None
    catch_z: float
    ply: str
    spawns: dict
    default_spawn: str
    label: str = ""
    """The web panel's map dropdown entry (the name without one)."""


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
        label="map3 楼梯井",
    ),
    # MetaCam map_2 indoor (spirula_ba_s2 step-50000, 2.98M gaussians).
    # Spawn = the gsplat-validated low-capture-band spot (map2_go2.json
    # initial_qpos, yaw -11.49); ground top -1.20 from the spawn-footprint
    # gaussian histogram -- see map2_scene.xml, recalibrate ground there
    # AND here together by render.
    "map2": Scene(
        xml=ASSETS / "mjcf" / "map2_scene.xml",
        collision=None,
        catch_z=-2.6,
        ply="map2_scene.ply",
        spawns={"low": ((3.21, -5.54, -0.755), -11.49)},
        default_spawn="low",
        label="map2 室内",
    ),
    # flat smoke terrain: procedural ground plane + go2 (build_flat_model),
    # no walls and no catch cliff -- the emu bench's long-run scene. The
    # 20 m plane bounds the walk far before integrator drift matters.
    "flat": Scene(
        xml=None,
        collision=None,
        catch_z=-1.0,
        ply="",
        spawns={"origin": ((0.0, 0.0, 0.32), 0.0)},
        default_spawn="origin",
        label="flat 平地",
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
    # Actuated joints: PIE order for go2, model order for the other robots.
    # joint_qpos_adr / joint_dof_adr / actuator_ids follow this order.
    robot: Robot
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    ctrl_default: np.ndarray   # standing ctrl: default angle, 0 for wheels
    scene_name: str = "map3"   # the scene it was built in (SCENES key)


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


def _load_robot_spec(robot: Robot) -> mujoco.MjSpec:
    # The MJCFs carry a meshdir="assets" placeholder; the real meshes are the
    # shared decimated MSH set (model/ bundle), spliced in here. go2_pie.xml
    # also names .obj files (path + extension).
    if robot.pie_plant:
        xml = GO2_PIE_XML.read_text()
        xml = re.sub(r'file="([a-z_0-9]+)\.obj"', r'file="\1.msh"', xml)
    else:
        xml = robot.mjcf.read_text()
    xml = xml.replace('meshdir="assets"', f'meshdir="{_meshdir(robot)}"')
    spec = mujoco.MjSpec.from_string(xml)
    if robot.pie_plant:
        _configure_robot_spec(spec)
    return spec


def _add_depth_camera_and_shell(base, robot: Robot) -> None:
    base.add_camera(
        name=DEPTH_CAMERA_NAME,
        pos=robot.camera_pos,
        quat=robot.camera_quat,
        fovy=C.DEPTH_FOVY_DEG,
    )
    # Camera shell visual, same dims/pose as PIE sim2sim (contype 0).
    base.add_geom(
        name=SHELL_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=np.add(robot.camera_pos, SHELL_FROM_DEPTH_CAMERA),
        quat=robot.camera_quat,
        size=SHELL_HALF_SIZE,
        rgba=(0.0, 0.0, 0.0, 1.0),
        contype=0,
        conaffinity=0,
        density=0.0,
        group=2,
    )


NAVCORE_RGBA = (0.12, 0.12, 0.13, 1.0)


WEB_HIDDEN_GROUP = 6   # the web viewer draws geom groups 0-5 only


def place_depth_shell(model: mujoco.MjModel, robot: Robot, navcore: bool) -> None:
    """The depth camera's shell geom on `model` (visual only: contype 0, no
    mass): the robot's navcore unit (robot.navcore, behind its lenses, so a
    navcore camera does not see it), or PIE's shell at the robot's trained
    mount (navcore=False, or a robot without a navcore box)."""
    shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SHELL_GEOM_NAME)
    if shell < 0:
        return
    if navcore and robot.navcore is not None:
        pos, quat, half = robot.navcore[0], (1.0, 0.0, 0.0, 0.0), robot.navcore[1]
        rgba = NAVCORE_RGBA
    else:
        pos, quat, half = np.add(robot.camera_pos, SHELL_FROM_DEPTH_CAMERA), robot.camera_quat, SHELL_HALF_SIZE
        rgba = (0.0, 0.0, 0.0, 1.0)
    model.geom_pos[shell] = pos
    model.geom_quat[shell] = quat
    model.geom_size[shell] = half
    model.geom_rgba[shell] = rgba
    model.geom_aabb[shell] = (0.0, 0.0, 0.0, *half)
    model.geom_rbound[shell] = float(np.linalg.norm(half))


def web_view_model(model: mujoco.MjModel, robot: Robot) -> mujoco.MjModel:
    """Copy of `model` for the web view: the depth camera shell redrawn as the
    robot's navcore unit (robot.navcore), and the visual meshes the robot no
    longer carries (robot.web_hidden_meshes, e.g. B2's LiDAR) moved to a group
    the web viewer never draws. Physics and the depth camera keep `model`, so
    the policy's depth input does not change."""
    web = copy.deepcopy(model)
    place_depth_shell(web, robot, navcore=True)
    hidden = {mujoco.mj_name2id(web, mujoco.mjtObj.mjOBJ_MESH, m) for m in robot.web_hidden_meshes} - {-1}
    base = mujoco.mj_name2id(web, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    for g in range(web.ngeom):
        if web.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if web.geom_dataid[g] in hidden:
            web.geom_group[g] = WEB_HIDDEN_GROUP
        elif robot.web_hidden_boxes and web.geom_bodyid[g] == base:
            _hide_mesh_pieces(web, g, robot.web_hidden_boxes)
    return web


_HIDDEN_FACES: dict = {}   # (mesh vertex / face counts, geom pose, boxes) -> faces to drop


def _mesh_pieces(faces: np.ndarray, nv: int) -> np.ndarray:
    """Per-vertex label of its piece (vertices connected through faces): hooking
    and pointer jumping on numpy arrays (B2-W's trunk: 183k vertices)."""
    label = np.arange(nv)
    u = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    v = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    while True:
        lu, lv = label[u], label[v]
        moved = lu != lv
        if not moved.any():
            return label
        lo, hi = np.minimum(lu[moved], lv[moved]), np.maximum(lu[moved], lv[moved])
        np.minimum.at(label, hi, lo)        # hook the larger root onto the smaller
        while True:                         # jump to the roots
            nxt = label[label]
            if np.array_equal(nxt, label):
                break
            label = nxt


def _hide_mesh_pieces(model: mujoco.MjModel, geom: int, boxes) -> None:
    """Collapse the faces of every separate piece (faces connected through shared
    vertices) of `geom`'s mesh that lies inside one of `boxes` ((lo, hi) in the
    geom's body frame). Edits `model`'s mesh data: a web view copy only."""
    mesh = model.geom_dataid[geom]
    v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    faces = model.mesh_face[f0:f0 + nf]
    key = (int(nv), int(nf), tuple(np.round(model.geom_pos[geom], 6)),
           tuple(np.round(model.geom_quat[geom], 6)), tuple(map(tuple, map(np.ravel, boxes))))
    if key not in _HIDDEN_FACES:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
        verts = model.mesh_vert[v0:v0 + nv] @ rot.reshape(3, 3).T + model.geom_pos[geom]
        piece = _mesh_pieces(faces, nv)
        order = np.argsort(piece, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(piece[order]) != 0])
        lo = np.minimum.reduceat(verts[order], starts)
        hi = np.maximum.reduceat(verts[order], starts)
        inside = np.zeros(len(starts), dtype=bool)
        for box_lo, box_hi in boxes:
            inside |= np.all(lo >= box_lo, axis=1) & np.all(hi <= box_hi, axis=1)
        drop = np.isin(piece, piece[order][starts][inside])
        _HIDDEN_FACES[key] = drop[faces[:, 0]]
    gone = _HIDDEN_FACES[key]
    model.mesh_face[f0:f0 + nf][gone] = faces[gone][:, :1]   # degenerate: all three corners one vertex


def _apply_plant_options(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = C.PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 500


def _joint_names(model: mujoco.MjModel, robot: Robot) -> tuple[str, ...]:
    """PIE's actuated order for go2; model order (= the training joint
    order) for the training MJCFs, every hinge joint actuated."""
    if robot.pie_plant:
        return C.ACTUATED_JOINT_NAMES
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)


def _bindings(model: mujoco.MjModel, shell_id: int, robot: Robot,
              scene_name: str = "map3") -> Bindings:
    def id_of(obj_type, name):
        i = mujoco.mj_name2id(model, obj_type, name)
        if i < 0:
            raise ValueError(f"model is missing {obj_type.name} {name!r}")
        return int(i)

    joint_names = _joint_names(model, robot)
    joint_qpos_adr = np.empty(len(joint_names), dtype=np.int32)
    joint_dof_adr = np.empty(len(joint_names), dtype=np.int32)
    actuator_ids = np.empty(len(joint_names), dtype=np.int32)
    for index, name in enumerate(joint_names):
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
    default_joint_pos = (np.asarray(C.DEFAULT_JOINT_POS, np.float64)
                         if robot.pie_plant
                         else robot.default_joint_pos(joint_names))
    # Velocity actuators (wheels: gain kv, no position bias) stand at 0.
    position = model.actuator_biasprm[actuator_ids, 1] != 0.0
    ctrl_default = np.where(position, default_joint_pos, 0.0)
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
        robot=robot,
        joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        ctrl_default=ctrl_default,
        scene_name=scene_name,
    )


def _validate(model: mujoco.MjModel, b: Bindings) -> None:
    if not np.isclose(float(model.cam_fovy[b.depth_camera_id]), C.DEPTH_FOVY_DEG,
                      atol=1e-10):
        raise ValueError("depth camera fovy drifted from contract")
    if float(model.opt.timestep) != C.PHYSICS_DT:
        raise ValueError("scene option timestep is not PIE's 0.005")
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError("scene option integrator is not ImplicitFast")
    if not b.robot.pie_plant:
        # Training MJCF: its own actuators, one per hinge joint.
        n = len(b.joint_names)
        if (model.nq, model.nv, model.nu) != (7 + n, 6 + n, n):
            raise ValueError(f"{b.robot.name}: nq/nv/nu="
                             f"{(model.nq, model.nv, model.nu)} for {n} joints")
        return
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


def _compile(scene_spec: mujoco.MjSpec, with_shell: bool, robot: Robot,
             scene_name: str = "map3"):
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
        robot,
        scene_name,
    )
    _validate(model, b)
    return model, b


def build_model(scene_name: str = "map3",
                robot: str = DEFAULT_ROBOT) -> tuple[mujoco.MjModel, Bindings]:
    """Scene <name> + robot <robot> (sim/robots.py; go2 = training go2 +
    PIE profile) -> (model, bindings). Sets the active scene for
    reset_to_spawn's z-snapping."""
    global _ACTIVE
    sc = get_scene(scene_name)
    _ACTIVE = sc
    if sc.xml is None:
        return build_flat_model(robot)
    rb = get_robot(robot)
    scene = mujoco.MjSpec.from_file(
        str(_preprocess_scene(sc.xml, ASSETS / ".compose"))
    )
    # Training terrain friction on the whole map (boxes + catch floor).
    for geom in scene.geoms:
        geom.friction[:] = TERRAIN_FRICTION

    robot_spec = _load_robot_spec(rb)
    yaw = np.radians(SPAWN_YAW_DEG)
    frame = scene.worldbody.add_frame(
        pos=list(SPAWN_XYZ), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    )
    base = next(b for b in robot_spec.worldbody.bodies if b.name == "base_link")
    frame.attach_body(base, "")
    base_in_scene = next(b for b in scene.worldbody.bodies if b.name == "base_link")

    _add_depth_camera_and_shell(base_in_scene, rb)
    rig = np.add(rb.camera_pos, RIG_FROM_DEPTH_CAMERA)
    for name, y in RIG_Y.items():
        cam = base_in_scene.add_camera(
            name=name, pos=[rig[0], y, rig[2]], quat=RIG_QUAT_WXYZ
        )
        cam.fovy = RIG_FOVY
    from hil import CONTRACT_FOVY_DEG  # the looper contract's single source

    base_in_scene.add_camera(
        name=CONTRACT_DEPTH_CAMERA,
        pos=[rig[0], RIG_Y["infra1"], rig[2]],
        quat=RIG_QUAT_WXYZ,
        fovy=CONTRACT_FOVY_DEG,
    )
    _apply_plant_options(scene)

    model, b = _compile(scene, with_shell=True, robot=rb, scene_name=scene_name)

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


def build_map3_model(robot: str = DEFAULT_ROBOT) -> tuple[mujoco.MjModel, Bindings]:
    """Back-compat alias: build_model("map3")."""
    return build_model("map3", robot)


def build_flat_model(robot: str = DEFAULT_ROBOT) -> tuple[mujoco.MjModel, Bindings]:
    """Ground plane + robot (upstream sim2sim 'flat' analogue)."""
    rb = get_robot(robot)
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
    robot_spec = _load_robot_spec(rb)
    frame = scene.worldbody.add_frame(pos=(0.0, 0.0, 0.0))
    base = next(b for b in robot_spec.worldbody.bodies if b.name == "base_link")
    frame.attach_body(base, "")
    base_in_scene = next(b for b in scene.worldbody.bodies if b.name == "base_link")
    _add_depth_camera_and_shell(base_in_scene, rb)
    _apply_plant_options(scene)
    return _compile(scene, with_shell=True, robot=rb, scene_name="flat")


def reset_to_spawn(model: mujoco.MjModel, data: mujoco.MjData, b: Bindings,
                   settle: bool = True, pos: tuple | None = None,
                   yaw_deg: float | None = None) -> None:
    """Drop the dog at the scene spawn (map3), origin+0.32 (flat), or an
    explicit pos/yaw override (e.g. the R2b stairs-down start on
    landing_F4_east at (-8.75, 5.7) facing -y = yaw 270). Heights are
    go2's; other robots stand their own spawn_drop above the ground. The
    scene is the one the model was built in (b.scene_name)."""
    mujoco.mj_resetData(model, data)
    root = b.root_qpos_adr
    drop = b.robot.spawn_drop
    sc = SCENES.get(b.scene_name, _ACTIVE)
    if pos is None and sc.collision is None and sc.xml is not None:
        # map2 and the like: their default spawn (a validated standing pose)
        pos, yaw_deg = sc.spawns[sc.default_spawn]
    if pos is not None:
        if sc.collision is not None:
            boxes = load_boxes(str(sc.collision))
            z = ground_top_at(boxes, *pos[:2], pos[2], catch_z=sc.catch_z) + drop
        else:
            # scenes without a collision fragment: the hint IS a validated
            # standing pose (e.g. map2's gsplat initial_qpos) for go2
            z = pos[2] + drop - SPAWN_DROP
        yaw = np.radians(yaw_deg if yaw_deg is not None else 0.0)
        data.qpos[root:root + 3] = (pos[0], pos[1], z)
    elif sc.collision is not None:  # map3: 123 collision boxes + floor -> find slab top
        boxes = load_boxes(str(sc.collision))
        z = ground_top_at(boxes, *SPAWN_XYZ[:2], SPAWN_XYZ[2], catch_z=sc.catch_z) + drop
        yaw = np.radians(SPAWN_YAW_DEG)
        data.qpos[root:root + 3] = (SPAWN_XYZ[0], SPAWN_XYZ[1], z)
    else:
        z, yaw = 0.32 + drop - SPAWN_DROP, 0.0
        data.qpos[root:root + 3] = (0.0, 0.0, z)
    if pos is None and yaw_deg is not None:
        yaw = np.radians(yaw_deg)
    data.qpos[root + 3:root + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    data.qpos[b.joint_qpos_adr] = b.default_joint_pos
    data.qvel[:] = 0.0
    data.ctrl[b.actuator_ids] = b.ctrl_default
    if settle:
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise FloatingPointError("state not finite after spawn reset")


def main() -> int:
    robot = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROBOT
    for name, fn in (("flat", build_flat_model), ("map3", build_map3_model)):
        model, b = fn(robot)
        print(f"[{robot} {name}] nq/nv/nu={model.nq}/{model.nv}/{model.nu} "
              f"dt={model.opt.timestep} integ={model.opt.integrator} "
              f"cam_fovy={model.cam_fovy[b.depth_camera_id]:.4f} "
              f"geoms={model.ngeom} timestep_ok=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
