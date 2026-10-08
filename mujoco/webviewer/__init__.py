"""Headless web viewer for the MuJoCo scene (vendored).

Vendored from mujocolab/mjviser@ae0b219 (Apache-2.0), files scene.py and
conversions.py; the full-UI viewer.py / CLI __main__.py were dropped (hil
drives ViserMujocoScene directly). Local changes on top of upstream:

  conversions.py
    * int() casts on geom_type-vs-mjtGeom enum comparisons -- numpy scalar
      == pybind enum falls back to identity and is always False, which made
      any mesh-bearing model crash the scene build (upstream PR #32).
    * _apply_flat_color: glTF transparency is MATERIAL-level -- attach a
      BLEND PBRMaterial for geom_rgba alpha < 1 (three.js ignores
      vertex-color alpha).
  scene.py
    * grid shadow_opacity 0.2 -> 0 and every receive_shadow=0.2 -> False
      (no shadows anywhere; they are rendered client-side anyway).
    * _add_fixed_geometry: translucent geoms are emitted as SEPARATE
      per-panel meshes (a merged mesh self-overlaps under doubleSided BLEND
      and reads near-opaque -- three.js can only sort per OBJECT), while
      opaque geoms in the same group still merge into ONE mesh.
"""

from webviewer.scene import ViserMujocoScene

__all__ = ["ViserMujocoScene"]
