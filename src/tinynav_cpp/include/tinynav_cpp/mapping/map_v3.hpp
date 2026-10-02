#ifndef TINYNAV_CPP__MAPPING__MAP_V3_HPP_
#define TINYNAV_CPP__MAPPING__MAP_V3_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "tinynav_cpp/mapping/map_v2.hpp"

struct sqlite3;
struct sqlite3_stmt;

namespace tinynav::mapping
{

// Map format v3: the whole map in one SQLite file (writer:
// tools/mapio/sqlite_writer.py, migrator: tools/migrate_map_to_v3.py).
// Same consumption surface as MapV2 — timestamps/poses/vlad_centres/
// vlad_descriptors are eager, per-keyframe features and depth are borrowed
// zero-copy views — plus the optional SigLIP semantic block and the meta
// table as string pairs.
//
// Zero-copy here means views over sqlite BLOB pages: every arrays row keeps
// its own prepared statement alive inside MapV3 (a BLOB pointer is only
// valid until its statement is stepped/reset/finalized, so a plain scan
// would dangle earlier rows). The views borrow the MapV3 owner exactly like
// the MapV2 mmap views borrow the mapping — the owner must outlive them
// (true at the only call site, where they die at the end of the reloc
// attempt). The sqlite file is opened read-only.
struct MapV3
{
  std::vector<int64_t> timestamps;
  std::unordered_map<int64_t, Eigen::Matrix4d> poses;
  Eigen::MatrixXd vlad_centres;        // C x d
  VladIndex vlad_descriptors;          // N x d f32 row-major
  Eigen::MatrixXf semantic_embeddings; // N x 768, zero row = missing; empty when the
                                       // block is absent or holds no non-zero row
  std::map<std::string, std::string> meta;

  bool has_frame(int64_t ts) const;
  // Non-owning view into the sqlite BLOBs; false when ts is unknown.
  bool get_features(int64_t ts, MapV2Features & out) const;
  // u16-millimeter rows convert to an owning CV_32F HxW Mat (meters);
  // f32-meter rows are non-owning views. Empty Mat when ts is unknown.
  cv::Mat get_depth(int64_t ts) const;

  struct ArrayView
  {
    const uint8_t * data = nullptr;
    uint64_t bytes = 0;
    std::vector<int64_t> shape;
    int elem_size = 0;
    bool is_f32 = false;  // depth <f4 meters; everything else fixed-width
  };

  std::unordered_map<int64_t, ArrayView> feature_kpts_, feature_descps_, feature_mask_,
    depth_;
  std::unordered_map<int64_t, uint32_t> row_of_;
  std::vector<int64_t> feature_offsets_;  // N+1, derived from per-row feature counts
  int64_t desc_dim_ = 0;

  ~MapV3();
  MapV3() = default;
  MapV3(MapV3 && other) noexcept;
  MapV3 & operator=(MapV3 && other) noexcept;
  MapV3(const MapV3 &) = delete;
  MapV3 & operator=(const MapV3 &) = delete;

  sqlite3 * db_ = nullptr;
  std::vector<sqlite3_stmt *> keepalive_;
};

// Load a map-format-v3 sqlite file. Returns false with `error` set on any
// missing/unreadable required block (keyframes/vlad/depth/features) — the
// caller degrades relocalization, never crashes. A missing or all-zero
// semantic block is not an error: the map loads with semantic_embeddings
// left empty, same degrade-don't-crash contract as map_v2's sidecar.
bool load_map_v3(const std::string & path, MapV3 & out, std::string & error);

}  // namespace tinynav::mapping

#endif  // TINYNAV_CPP__MAPPING__MAP_V3_HPP_
