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
    * _can_merge_vertices: vectorised one-colour check (a set of per-vertex
      tuples took ~0.5 s per robot merge).
  scene.py
    * grid shadow_opacity 0.2 -> 0 and every receive_shadow=0.2 -> False
      (no shadows anywhere; they are rendered client-side anyway).
    * no "Track camera" checkbox when camera_tracking_enabled is False at
      create_visualization_gui() time: every host turns it off and frames the
      camera itself, and checking it shifted the scene under that framing.
    * _add_fixed_geometry: translucent geoms are emitted as SEPARATE
      per-panel meshes (a merged mesh self-overlaps under doubleSided BLEND
      and reads near-opaque -- three.js can only sort per OBJECT), while
      opaque geoms in the same group still merge into ONE mesh.
    * mesh_cache / fill_mesh_cache(): merged body meshes reused by later
      scenes of the same model (view.py pre-merges every robot for switches).
    * robot meshes with lod="off": full detail at any distance.
    * frame_clients, and ONE connect hook per server calling the newest
      scene's: a rebuilt scene no longer re-frames open pages or stays alive.
    * scene_offset / scene_yaw: a host-set scene transform (viewer =
      Rz(yaw) (world + offset)); the offset wins over the built-in tracking
      -- view.py's Track robot.
    * add_camera_frustum() / show_camera_frustums: a model camera drawn as
      a pyramid riding its body -- the policy's depth camera.
"""

from webviewer.scene import ViserMujocoScene

__all__ = ["ViserMujocoScene"]
