#!/usr/bin/env python3
"""Parametric flat-floor stairwell plan for planning tests.

The physical world is a single flat floor.  What "stairs" means here is only
the PLAN VIEW of a switchback stairwell: a rectangular ring corridor around a
central well, with two long sides (the two flights) and two short sides (the
two landings).  There is no vertical step, no stair geometry and no climbing:
the dog drives the ring on flat ground, so the nav stack sees the same U/ring
route it would see upstairs/downstairs.

Geometry (all z=0 floor; walls are boxes standing on it):

    ┌──────────────────────────────────────────┐  outer wall
    │  ┌────────────────────────────────────┐  │
    │  │                                    │  │
    │  │            well (solid)            │  │  inner well wall
    │  │                                    │  │
    │  └────────────────────────────────────┘  │
    │                                          │
    │   platform W                 platform E  │
    └──────────────────────────────────────────┘
          flight N (top) / flight S (bottom)

`flight_len` is the long-side length (the well's x extent), `flight_width` the
ring corridor width, `well_width` the well's y extent, `platform_len` the short
side's depth.  `platform_width` is the short side's free width and is fixed by
the ring: `2*flight_width + well_width`; passing it explicitly is validated.

The scene is a closed ring: the dog can keep circulating forever.  Four
waypoints (W, N, E, S) lie at the midpoints of the four sides; the default
spawn is W facing +y so the loop order W -> N -> E -> S -> W starts cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # nav-side probes only need the geometry helpers
    mujoco = None

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_CONFIG = ASSETS / "floorplan" / "default.yaml"

# terrain friction used by the map3 plant (training value); the go2 feet are
# 0.6 and MuJoCo combines friction element-wise by max.
TERRAIN_FRICTION = (0.6, 0.005, 0.0001)

SPAWN_DROP = 0.30


@dataclass(frozen=True)
class FloorPlanParams:
    """One flat stairwell plan.  Dimensions in metres.

    flight_len    long-side corridor length (= well x extent)
    flight_width  ring corridor width (long sides / flights)
    well_width    well y extent (the gap between the two flights)
    platform_len  short-side corridor depth (the two landings)
    platform_width  short-side free width; fixed by the ring, optional input
    wall_height / wall_thickness  outer and inner well walls
    floor_thickness  floor slab below z=0
    checker       checker texture on the floor (feature-rich for mapping)
    """

    flight_len: float = 4.0
    flight_width: float = 1.0
    well_width: float = 1.0
    platform_len: float = 1.5
    platform_width: float | None = None
    wall_height: float = 1.2
    wall_thickness: float = 0.06
    floor_thickness: float = 0.10
    checker: bool = True

    def __post_init__(self) -> None:
        vals = {
            "flight_len": self.flight_len,
            "flight_width": self.flight_width,
            "well_width": self.well_width,
            "platform_len": self.platform_len,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
            "floor_thickness": self.floor_thickness,
        }
        for name, v in vals.items():
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be positive, got {v}")
        if self.flight_width <= 2.0 * self.wall_thickness:
            raise ValueError("flight_width must exceed twice wall_thickness")
        if self.well_width <= 2.0 * self.wall_thickness:
            raise ValueError("well_width must exceed twice wall_thickness")
        if self.flight_len <= 2.0 * self.wall_thickness:
            raise ValueError("flight_len must exceed twice wall_thickness")
        derived = self.ring_width
        if self.platform_width is not None and not np.isclose(
                float(self.platform_width), derived, atol=1e-9):
            raise ValueError(
                f"platform_width={self.platform_width} is fixed by the ring: "
                f"2*flight_width + well_width = {derived}")
        if self.flight_width < 0.7:
            # a go2 is ~0.6 m wide; narrower corridors are not a useful test
            raise ValueError(
                f"flight_width={self.flight_width} is too narrow for the "
                "go2 (need >= 0.7 m)")

    # -- derived plan-face dimensions -------------------------------------
    @property
    def ring_width(self) -> float:
        """Short-side free width (platform width)."""
        return self.well_width + 2.0 * self.flight_width

    @property
    def free_len_x(self) -> float:
        """Free ring extent along x (inside the outer walls)."""
        return self.flight_len + 2.0 * self.platform_len

    @property
    def free_len_y(self) -> float:
        """Free ring extent along y (inside the outer walls)."""
        return self.ring_width

    @property
    def outer_len_x(self) -> float:
        return self.free_len_x + 2.0 * self.wall_thickness

    @property
    def outer_len_y(self) -> float:
        return self.free_len_y + 2.0 * self.wall_thickness

    def summary(self) -> str:
        return (
            f"floorplan({self.flight_len:.2f}x{self.flight_width:.2f} "
            f"flights, well {self.well_width:.2f}, platform "
            f"{self.platform_len:.2f}x{self.ring_width:.2f}, "
            f"walls {self.wall_height:.2f}, outer "
            f"{self.outer_len_x:.2f}x{self.outer_len_y:.2f})")


def load_params(path: str | Path | None = None) -> FloorPlanParams:
    """YAML -> FloorPlanParams; unknown keys are rejected loudly."""
    import yaml

    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_file():
        return FloorPlanParams()
    raw = yaml.safe_load(p.read_text()) or {}
    known = {f.name for f in fields(FloorPlanParams)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{p}: unknown floorplan keys {unknown}")
    return FloorPlanParams(**raw)


def _box(name: str, pos, half, rgba, material: str | None = None):
    return {
        "name": name,
        "pos": np.asarray(pos, float),
        "half": np.asarray(half, float),
        "rgba": tuple(float(v) for v in rgba),
        "material": material,
    }


def box_specs(params: FloorPlanParams) -> list[dict]:
    """Every static box: floor, outer walls, inner well walls, well block.

    The well is a solid block whose perimeter carries the inner wall boxes:
    the block fills the well inside the walls, so the robot can never enter
    and the planner sees an unambiguous obstacle.
    """
    L = params.flight_len
    Wf = params.flight_width
    Ww = params.well_width
    P = params.platform_len
    t = params.wall_thickness
    h = params.wall_height
    Lx = params.free_len_x
    Ly = params.free_len_y
    ft = params.floor_thickness

    outer = (0.30, 0.37, 0.47, 1.0)
    inner = (0.78, 0.45, 0.22, 1.0)
    well = (0.16, 0.17, 0.19, 1.0)
    floor = (0.64, 0.64, 0.61, 1.0)

    boxes = [
        # floor slab: top at z=0, extends under the outer walls
        _box("floor", (0.0, 0.0, -ft / 2.0),
             (Lx / 2.0 + t, Ly / 2.0 + t, ft / 2.0), floor, "floor"),
        # outer walls, outside the free region; inner faces at +/-Lx/2, +/-Ly/2
        _box("outer_w", (-(Lx / 2.0 + t / 2.0), 0.0, h / 2.0),
             (t / 2.0, Ly / 2.0 + t, h / 2.0), outer, "outer_wall"),
        _box("outer_e", (+(Lx / 2.0 + t / 2.0), 0.0, h / 2.0),
             (t / 2.0, Ly / 2.0 + t, h / 2.0), outer, "outer_wall"),
        _box("outer_s", (0.0, -(Ly / 2.0 + t / 2.0), h / 2.0),
             (Lx / 2.0 + t, t / 2.0, h / 2.0), outer, "outer_wall"),
        _box("outer_n", (0.0, +(Ly / 2.0 + t / 2.0), h / 2.0),
             (Lx / 2.0 + t, t / 2.0, h / 2.0), outer, "outer_wall"),
        # inner well walls, thickness inside the well boundary so the ring
        # corridor keeps its nominal width
        _box("well_w", (-(L / 2.0 - t / 2.0), 0.0, h / 2.0),
             (t / 2.0, Ww / 2.0, h / 2.0), inner, "inner_wall"),
        _box("well_e", (+(L / 2.0 - t / 2.0), 0.0, h / 2.0),
             (t / 2.0, Ww / 2.0, h / 2.0), inner, "inner_wall"),
        _box("well_s", (0.0, -(Ww / 2.0 - t / 2.0), h / 2.0),
             (L / 2.0, t / 2.0, h / 2.0), inner, "inner_wall"),
        _box("well_n", (0.0, +(Ww / 2.0 - t / 2.0), h / 2.0),
             (L / 2.0, t / 2.0, h / 2.0), inner, "inner_wall"),
        # solid well core
        _box("well_core", (0.0, 0.0, h / 2.0),
             (L / 2.0 - t, Ww / 2.0 - t, h / 2.0), well, "well"),
    ]
    return boxes


def floorplan_boxes(params: FloorPlanParams):
    """(name, pos[3], half[3]) for ground-top queries / reset."""
    return [(b["name"], b["pos"], b["half"]) for b in box_specs(params)]


def build_floorplan_spec(params: FloorPlanParams | None = None):
    """Static flat-floor plan as an MjSpec (no robot, no cameras).

    Textures/materials come from an XML asset block: MuJoCo 3.14's MjSpec
    `add_texture` path compiles with empty texture data here (the builtin
    checker never reaches the renderer), while the XML asset path does.
    Geometry is still built with the typed MjSpec API.
    """
    p = params or FloorPlanParams()
    if mujoco is None:
        raise ImportError("mujoco is required to build the floorplan MjSpec")
    if p.checker:
        assets = (
            '<texture name="floor_grid" type="2d" builtin="checker" '
            'rgb1="0.58 0.58 0.55" rgb2="0.90 0.90 0.86" '
            'width="512" height="512"/>'
            '<material name="floor" texture="floor_grid" texrepeat="10 10" '
            'rgba="1 1 1 1"/>'
        )
    else:
        assets = '<material name="floor" rgba="0.70 0.70 0.67 1"/>'
    assets += (
        '<material name="outer_wall" rgba="0.42 0.50 0.62 1" '
        'specular="0.1" shininess="0.1"/>'
        '<material name="inner_wall" rgba="0.85 0.50 0.25 1" '
        'specular="0.1" shininess="0.1"/>'
        '<material name="well" rgba="0.20 0.21 0.24 1" '
        'specular="0.05" shininess="0.05"/>'
    )
    try:
        from sim import floorplan_textures as _fptex
    except ImportError:  # direct in-tree import of this module
        import floorplan_textures as _fptex
    tex_frag, tex_files = _fptex.material_assets()
    assets += tex_frag
    floor_frag, floor_geoms = _fptex.floor_material_assets(p)
    assets += floor_frag
    spec = mujoco.MjSpec.from_string(
        "<mujoco model='floorplan'><asset>" + assets +
        "</asset><worldbody/></mujoco>")
    for fname, blob in tex_files.items():
        spec.assets[fname] = blob
    # The contract cameras sit inside a 1 m corridor; lift the headlight and
    # the sun so the native stereo RGB is exposed like the splat scenes.
    spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse = (0.80, 0.80, 0.80)
    spec.visual.headlight.specular = (0.10, 0.10, 0.10)

    tex_keys = {seg for seg, _, _, _ in _fptex.SEGMENTS}
    for b in box_specs(p):
        if b["name"] == "floor":
            # seam underlay: the tinted region floors meet in hairline cracks
            # a foot can catch and fall through (watched it happen); keep the
            # original full slab 5 mm below so every seam lands on solid
            pos = list(b["pos"])
            pos[2] -= 0.005
            spec.worldbody.add_geom(
                name=b["name"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=b["half"].tolist(),
                rgba=b["rgba"],
                group=0,
                contype=1,
                conaffinity=1,
                friction=TERRAIN_FRICTION,
            )
            continue
        material = f"fp_{b['name']}" if b["name"] in tex_keys else b["material"]
        spec.worldbody.add_geom(
            name=b["name"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=b["pos"].tolist(),
            size=b["half"].tolist(),
            rgba=b["rgba"],
            material=material,
            group=0,
            contype=1,
            conaffinity=1,
            friction=TERRAIN_FRICTION,
        )
    for g in floor_geoms:
        spec.worldbody.add_geom(
            name=g["name"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(g["pos"]),
            size=list(g["half"]),
            material=g["material"],
            group=0,
            contype=1,
            conaffinity=1,
            friction=TERRAIN_FRICTION,
        )

    # one downward sun; the camera headlight supplies the rest
    spec.worldbody.add_light(
        name="sun", pos=(0.0, 0.0, 5.0), dir=(0.0, 0.0, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
        ambient=(0.30, 0.30, 0.30), diffuse=(0.80, 0.80, 0.80),
    )
    return spec


def floorplan_spawns(params: FloorPlanParams) -> dict[str, tuple]:
    """Named spawns at the four side midpoints.  Values: ((x,y,z), yaw_deg).

    W is the default spawn, facing +y so the ring order W->N->E->S->W starts
    in the intended direction.
    """
    p = params
    Lx, Ly = p.free_len_x, p.free_len_y
    P, Wf = p.platform_len, p.flight_width
    z = SPAWN_DROP
    w = (-Lx / 2.0 + P / 2.0, 0.0, z)
    n = (0.0, Ly / 2.0 - Wf / 2.0, z)
    e = (Lx / 2.0 - P / 2.0, 0.0, z)
    s = (0.0, -Ly / 2.0 + Wf / 2.0, z)
    return {
        "w": (w, 90.0),
        "n": (n, 0.0),
        "e": (e, 270.0),
        "s": (s, 180.0),
    }


def floorplan_waypoints(params: FloorPlanParams) -> list[dict]:
    """The four TASK points, all on the platforms.

    Each sits where a flight centreline meets its platform centreline --
    the four corners of the walking loop: WN/EN/ES/WS (west/east x
    north/south).  yaw_deg is the clockwise direction of travel through
    the corner; planning only consumes the position.
    """
    wx = -(params.free_len_x / 2.0 - params.platform_len / 2.0)
    cy = params.free_len_y / 2.0 - params.flight_width / 2.0
    return [
        {"name": "WN", "pos": (wx, +cy, SPAWN_DROP), "yaw_deg": 0.0},
        {"name": "EN", "pos": (-wx, +cy, SPAWN_DROP), "yaw_deg": 270.0},
        {"name": "ES", "pos": (-wx, -cy, SPAWN_DROP), "yaw_deg": 180.0},
        {"name": "WS", "pos": (wx, -cy, SPAWN_DROP), "yaw_deg": 90.0},
    ]


def floorplan_loop(params: FloorPlanParams, clockwise: bool = True) -> list[dict]:
    """The four platform task points in ring order; every one is a target.

    Clockwise:  WN -> EN -> ES -> WS   (ends one leg short of a full lap;
    the next lap's WN leg closes it along the west platform centreline)
    Counterclockwise:  WN -> WS -> ES -> EN

    Adjacent points are joined by the flight/platform centrelines, so a lap
    is four straight centreline legs -- no routing vias are needed.
    """
    wps = {w["name"]: w for w in floorplan_waypoints(params)}
    order = ("WN", "EN", "ES", "WS") if clockwise else ("WN", "WS", "ES", "EN")
    out = []
    for i, name in enumerate(order):
        nxt = order[(i + 1) % len(order)]
        a = wps[name]["pos"]
        b = wps[nxt]["pos"]
        # yaw = the outgoing leg, i.e. the stair the robot will walk next
        yaw = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
        out.append(dict(wps[name], yaw_deg=yaw, kind="target"))
    return out


__all__ = [
    "DEFAULT_CONFIG",
    "FloorPlanParams",
    "box_specs",
    "build_floorplan_spec",
    "floorplan_boxes",
    "floorplan_loop",
    "floorplan_spawns",
    "floorplan_waypoints",
    "load_params",
]
