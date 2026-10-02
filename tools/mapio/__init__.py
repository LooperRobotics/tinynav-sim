"""Map writer abstraction: the builder writes through MapWriter so the map
format (v2 npy or v3 single-file SQLite) is a replaceable detail.

Discipline (docs/migration-progress.md):
- finalize() numpy-readbacks everything it wrote and checks shape/dtype.
- Depth is accepted as f32 meters and stored u16 millimeters (0 = invalid),
  matching tools/export_map_v2.py and the C++ readers' f4/u2 sniffing.
- v2 keeps SigLIP semantic embeddings in a sidecar (semantic_embeddings.npy,
  row order = pose_timestamps, zero row = missing) + semantic_meta.json; v3
  stores them as semantic_embedding rows in the arrays table (absent = missing).
  The C++ reader contract ignores unknown files/extra rows, so old readers do
  not break; new readers must degrade to WARN on a missing/zero-norm block.
"""
from .sqlite_writer import SQLiteWriter, load_map_v3
from .writer import MapWriter, V2NpyWriter

__all__ = ["MapWriter", "V2NpyWriter", "SQLiteWriter", "load_map_v3"]
