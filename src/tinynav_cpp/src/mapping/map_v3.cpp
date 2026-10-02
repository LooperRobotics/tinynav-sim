// Map format v3 loader: reads the single-file SQLite map written by
// tools/mapio/sqlite_writer.py (migrator: tools/migrate_map_to_v3.py).
// Deliberately mirrors load_map_v2's contract: same consumption surface,
// same degrade-don't-crash rules — required blocks missing -> false + error,
// the optional semantic block missing/all-zero -> the map still loads.
#include "tinynav_cpp/mapping/map_v3.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cstring>
#include <sstream>

namespace tinynav::mapping
{

namespace
{

int elem_size_of(const std::string & dtype, bool & is_f32, std::string & error)
{
  is_f32 = false;
  if (dtype == "<f4") {
    is_f32 = true;
    return 4;
  }
  if (dtype == "<f8") return 8;
  if (dtype == "<u2") return 2;
  if (dtype == "<u1" || dtype == "|u1") return 1;
  if (dtype == "<i8") return 8;
  error = "arrays: unsupported dtype '" + dtype + "' (want <f4/<f8/<u2/<u1/<i8)";
  return 0;
}

bool parse_shape(const std::string & text, std::vector<int64_t> & out, std::string & error)
{
  out.clear();
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
    error = "arrays: bad shape text '" + text + "'";
    return false;
  }
  std::stringstream ss(text.substr(1, text.size() - 2));
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (!token.empty()) {
      out.push_back(std::stoll(token));
    }
  }
  return true;
}

uint64_t shape_count(const std::vector<int64_t> & shape)
{
  uint64_t total = 1;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

// Fetch one arrays row's BLOB through its own prepared statement, which is
// kept alive in `keepalive`: the pointer stays valid until that statement is
// stepped/reset/finalized, i.e. for as long as the MapV3 owner lives.
const uint8_t * fetch_blob(sqlite3 * db, int64_t ts, const char * name,
  sqlite3_stmt * & keep, uint64_t & bytes, std::string & error)
{
  if (sqlite3_prepare_v2(db, "SELECT data FROM arrays WHERE ts=?1 AND name=?2", -1, &keep,
      nullptr) != SQLITE_OK)
  {
    error = std::string("arrays: prepare failed: ") + sqlite3_errmsg(db);
    return nullptr;
  }
  sqlite3_bind_int64(keep, 1, ts);
  sqlite3_bind_text(keep, 2, name, -1, SQLITE_STATIC);
  if (sqlite3_step(keep) != SQLITE_ROW) {
    error = std::string("arrays: row vanished for ts=") + std::to_string(ts) + " name=" + name;
    sqlite3_finalize(keep);
    keep = nullptr;
    return nullptr;
  }
  const auto * ptr = static_cast<const uint8_t *>(sqlite3_column_blob(keep, 0));
  bytes = static_cast<uint64_t>(sqlite3_column_bytes(keep, 0));
  return ptr;
}

}  // namespace

MapV3::~MapV3()
{
  for (sqlite3_stmt * stmt : keepalive_) {
    if (stmt) {
      sqlite3_finalize(stmt);
    }
  }
  if (db_) {
    sqlite3_close(db_);
  }
}

MapV3::MapV3(MapV3 && other) noexcept
{
  *this = std::move(other);
}

MapV3 & MapV3::operator=(MapV3 && other) noexcept
{
  if (this != &other) {
    this->~MapV3();
    timestamps = std::move(other.timestamps);
    poses = std::move(other.poses);
    vlad_centres = std::move(other.vlad_centres);
    vlad_descriptors = std::move(other.vlad_descriptors);
    semantic_embeddings = std::move(other.semantic_embeddings);
    meta = std::move(other.meta);
    feature_kpts_ = std::move(other.feature_kpts_);
    feature_descps_ = std::move(other.feature_descps_);
    feature_mask_ = std::move(other.feature_mask_);
    depth_ = std::move(other.depth_);
    row_of_ = std::move(other.row_of_);
    feature_offsets_ = std::move(other.feature_offsets_);
    desc_dim_ = other.desc_dim_;
    db_ = other.db_;
    keepalive_ = std::move(other.keepalive_);
    other.db_ = nullptr;
  }
  return *this;
}

bool load_map_v3(const std::string & path, MapV3 & out, std::string & error)
{
  out = MapV3{};

  sqlite3 * db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    error = "cannot open " + path + ": " + (db ? sqlite3_errmsg(db) : "sqlite");
    if (db) {
      sqlite3_close(db);
    }
    return false;
  }
  out.db_ = db;  // owned from here; the dtor closes on every error path below

  sqlite3_stmt * stmt = nullptr;

  // --- meta + version gate -------------------------------------------------
  if (sqlite3_prepare_v2(db, "PRAGMA user_version", -1, &stmt, nullptr) != SQLITE_OK ||
    sqlite3_step(stmt) != SQLITE_ROW)
  {
    error = std::string("user_version unreadable: ") + sqlite3_errmsg(db);
    sqlite3_finalize(stmt);
    return false;
  }
  const int user_version = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  stmt = nullptr;
  if (user_version != 3) {
    error = "not a map v3 (user_version=" + std::to_string(user_version) + "): " + path;
    return false;
  }

  if (sqlite3_prepare_v2(db, "SELECT k, v FROM meta", -1, &stmt, nullptr) != SQLITE_OK) {
    error = std::string("meta: prepare failed: ") + sqlite3_errmsg(db);
    return false;
  }
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    out.meta[reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0))] =
      reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;
  const auto format = out.meta.find("format");
  if (format == out.meta.end() || format->second != "tinynav_map_v3") {
    error = "meta.format missing or not tinynav_map_v3";
    return false;
  }

  // --- keyframes -------------------------------------------------------------
  if (sqlite3_prepare_v2(db, "SELECT ts, pose FROM keyframes ORDER BY ts", -1, &stmt,
      nullptr) != SQLITE_OK)
  {
    error = std::string("keyframes: prepare failed: ") + sqlite3_errmsg(db);
    return false;
  }
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const int64_t ts = sqlite3_column_int64(stmt, 0);
    const auto * pose = static_cast<const double *>(sqlite3_column_blob(stmt, 1));
    if (!pose || sqlite3_column_bytes(stmt, 1) != 16 * static_cast<int>(sizeof(double))) {
      error = "keyframes: pose blob for ts=" + std::to_string(ts) + " is not 16 f64";
      sqlite3_finalize(stmt);
      return false;
    }
    Eigen::Matrix4d m;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        m(r, c) = pose[r * 4 + c];  // blob is row-major; Eigen is column-major
      }
    }
    out.timestamps.push_back(ts);
    out.poses[ts] = m;
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;
  const int64_t n = static_cast<int64_t>(out.timestamps.size());
  if (n == 0) {
    error = "map holds no keyframes";
    return false;
  }

  // --- arrays ----------------------------------------------------------------
  // Scan the row inventory first, then fetch each row's BLOB through its own
  // kept-alive statement (see fetch_blob) so the views stay zero-copy.
  struct RowRef
  {
    int64_t ts;
    std::string name;
    std::vector<int64_t> shape;
    int elem_size = 0;
    bool is_f32 = false;
  };
  std::vector<RowRef> rows;
  if (sqlite3_prepare_v2(db, "SELECT ts, name, dtype, shape FROM arrays", -1, &stmt,
      nullptr) != SQLITE_OK)
  {
    error = std::string("arrays: prepare failed: ") + sqlite3_errmsg(db);
    return false;
  }
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    RowRef ref;
    ref.ts = sqlite3_column_int64(stmt, 0);
    ref.name = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
    bool is_f32 = false;
    std::string dtype = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
    if (!parse_shape(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3)), ref.shape,
      error))
    {
      sqlite3_finalize(stmt);
      return false;
    }
    if ((ref.elem_size = elem_size_of(dtype, is_f32, error)) == 0) {
      sqlite3_finalize(stmt);
      return false;
    }
    ref.is_f32 = is_f32;
    rows.push_back(std::move(ref));
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  bool has_features_block = false;
  std::unordered_map<int64_t, Eigen::VectorXf> vlad_rows;
  std::unordered_map<int64_t, Eigen::VectorXf> semantic_rows;

  for (const RowRef & ref : rows) {
    MapV3::ArrayView view;
    view.shape = ref.shape;
    view.elem_size = ref.elem_size;
    view.is_f32 = ref.is_f32;
    sqlite3_stmt * keep = nullptr;
    view.data = fetch_blob(db, ref.ts, ref.name.c_str(), keep, view.bytes, error);
    out.keepalive_.push_back(keep);
    if (!view.data) {
      return false;
    }
    const uint64_t want_bytes = shape_count(ref.shape) * static_cast<uint64_t>(ref.elem_size);
    if (view.bytes != want_bytes) {
      error = "arrays: byte count mismatch for ts=" + std::to_string(ref.ts) + " name=" + ref.name;
      return false;
    }

    const auto & nm = ref.name;
    if (nm == "depth") {
      if (ref.shape.size() != 2) {
        error = "arrays: depth must be [H,W] per keyframe";
        return false;
      }
      out.depth_[ref.ts] = view;
    } else if (nm == "feature_kpts" || nm == "feature_descps" || nm == "feature_mask") {
      has_features_block = true;
      if (nm == "feature_kpts") {
        out.feature_kpts_[ref.ts] = view;
      } else if (nm == "feature_descps") {
        out.feature_descps_[ref.ts] = view;
      } else {
        out.feature_mask_[ref.ts] = view;
      }
    } else if (nm == "vlad_descriptor") {
      if (ref.shape.size() != 1 || ref.elem_size != 4) {
        error = "arrays: vlad_descriptor must be [d] <f4";
        return false;
      }
      Eigen::VectorXf v(static_cast<int64_t>(ref.shape[0]));
      std::memcpy(v.data(), view.data, view.bytes);
      vlad_rows[ref.ts] = std::move(v);
    } else if (nm == "semantic_embedding") {
      if (ref.shape.size() != 1 || ref.elem_size != 4) {
        error = "arrays: semantic_embedding must be [768] <f4";
        return false;
      }
      Eigen::VectorXf v(static_cast<int64_t>(ref.shape[0]));
      std::memcpy(v.data(), view.data, view.bytes);
      semantic_rows[ref.ts] = std::move(v);
    }
    // unknown names are ignored — forward compat with future per-keyframe arrays
  }

  // --- required-block contract (mirrors load_map_v2) -------------------------
  for (int64_t i = 0; i < n; ++i) {
    const int64_t ts = out.timestamps[static_cast<size_t>(i)];
    if (out.depth_.find(ts) == out.depth_.end()) {
      error = "depth missing for keyframe ts=" + std::to_string(ts);
      return false;
    }
    const auto vit = vlad_rows.find(ts);
    if (vit == vlad_rows.end()) {
      error = "vlad_descriptor missing for keyframe ts=" + std::to_string(ts);
      return false;
    }
    const int64_t d = vit->second.size();
    if (i == 0) {
      out.vlad_descriptors.resize(n, d);
    } else if (out.vlad_descriptors.cols() != d) {
      error = "vlad_descriptor dim mismatch across keyframes";
      return false;
    }
    out.vlad_descriptors.row(static_cast<int>(i)) = vit->second.transpose();
    out.row_of_[ts] = static_cast<uint32_t>(i);
  }
  if (!has_features_block) {
    error = "features block missing (no feature_kpts/feature_descps/feature_mask rows)";
    return false;
  }

  out.feature_offsets_.assign(static_cast<size_t>(n) + 1, 0);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t ts = out.timestamps[static_cast<size_t>(i)];
    const auto kf = out.feature_kpts_.find(ts);
    const auto df = out.feature_descps_.find(ts);
    const auto mf = out.feature_mask_.find(ts);
    const bool any = kf != out.feature_kpts_.end();
    if (any != (df != out.feature_descps_.end()) || any != (mf != out.feature_mask_.end())) {
      error = "features incomplete for ts=" + std::to_string(ts) +
              " (kpts/descps/mask must come as a set)";
      return false;
    }
    int64_t rows_here = 0;
    if (any) {
      if (kf->second.shape.size() != 2 || kf->second.shape[1] != 2) {
        error = "feature_kpts must be [M,2]";
        return false;
      }
      rows_here = kf->second.shape[0];
      if (df->second.shape.size() != 2 || df->second.shape[0] != rows_here) {
        error = "feature_descps rows != feature_kpts rows";
        return false;
      }
      if (mf->second.shape != std::vector<int64_t>{rows_here}) {
        error = "feature_mask count != feature_kpts rows";
        return false;
      }
      if (i == 0 || out.desc_dim_ == 0) {
        out.desc_dim_ = df->second.shape[1];
      } else if (out.desc_dim_ != df->second.shape[1]) {
        error = "feature_descps dim mismatch across keyframes";
        return false;
      }
    }
    out.feature_offsets_[static_cast<size_t>(i) + 1] =
      out.feature_offsets_[static_cast<size_t>(i)] + rows_here;
  }
  if (out.desc_dim_ == 0) {
    error = "features block holds no rows";
    return false;
  }

  // --- vlad_centres blob -------------------------------------------------------
  const auto dtype_it = out.meta.find("blob.vlad_centres.dtype");
  const auto shape_it = out.meta.find("blob.vlad_centres.shape");
  if (dtype_it == out.meta.end() || shape_it == out.meta.end()) {
    error = "blobs: vlad_centres dtype/shape metadata missing";
    return false;
  }
  bool centres_f32 = false;
  const int centre_elem = elem_size_of(dtype_it->second, centres_f32, error);
  std::vector<int64_t> centre_shape;
  if (centre_elem == 0 || !parse_shape(shape_it->second, centre_shape, error) ||
    centre_shape.size() != 2)
  {
    error = "blobs: vlad_centres " + error;
    return false;
  }
  if (sqlite3_prepare_v2(db, "SELECT data FROM blobs WHERE name='vlad_centres'", -1, &stmt,
      nullptr) != SQLITE_OK ||
    sqlite3_step(stmt) != SQLITE_ROW)
  {
    error = std::string("blobs: vlad_centres missing: ") + sqlite3_errmsg(db);
    sqlite3_finalize(stmt);
    return false;
  }
  {
    const auto * bytes = static_cast<const uint8_t *>(sqlite3_column_blob(stmt, 0));
    const uint64_t got = static_cast<uint64_t>(sqlite3_column_bytes(stmt, 0));
    const uint64_t want = shape_count(centre_shape) * static_cast<uint64_t>(centre_elem);
    if (!bytes || got != want) {
      error = "blobs: vlad_centres byte count mismatch";
      sqlite3_finalize(stmt);
      return false;
    }
    const int64_t c = centre_shape[0], dim = centre_shape[1];
    out.vlad_centres.resize(c, dim);
    if (centre_elem == 8) {
      std::memcpy(out.vlad_centres.data(), bytes, want);
    } else {
      const auto * src = reinterpret_cast<const float *>(bytes);
      for (int64_t r = 0; r < c; ++r) {
        for (int64_t col = 0; col < dim; ++col) {
          out.vlad_centres(r, col) = src[static_cast<size_t>(r * dim + col)];
        }
      }
    }
  }
  sqlite3_finalize(stmt);
  stmt = nullptr;

  // --- optional semantic block -------------------------------------------------
  // Missing or all-zero is fine (degrade, don't crash) — consumers treat an
  // empty matrix as "no semantic index", like the absent v2 sidecar.
  if (!semantic_rows.empty()) {
    Eigen::MatrixXf sem = Eigen::MatrixXf::Zero(n, 768);
    int64_t non_zero = 0;
    for (int64_t i = 0; i < n; ++i) {
      const auto it = semantic_rows.find(out.timestamps[static_cast<size_t>(i)]);
      if (it != semantic_rows.end() && it->second.size() == 768) {
        sem.row(static_cast<int>(i)) = it->second.transpose();
        if (!it->second.isZero()) {
          ++non_zero;
        }
      }
    }
    if (non_zero > 0) {
      out.semantic_embeddings = std::move(sem);
    }
  }

  return true;
}

bool MapV3::has_frame(const int64_t ts) const
{
  return row_of_.find(ts) != row_of_.end() && depth_.find(ts) != depth_.end();
}

bool MapV3::get_features(const int64_t ts, MapV2Features & out) const
{
  out = MapV2Features{};
  const auto it = row_of_.find(ts);
  if (it == row_of_.end()) {
    return false;
  }
  const size_t i = it->second;
  const int64_t begin = feature_offsets_[i];
  const int64_t end = feature_offsets_[i + 1];
  const int rows = static_cast<int>(end - begin);
  if (rows <= 0) {
    return true;  // keyframe stored no features — caller skips empty kpts
  }
  const auto kf = feature_kpts_.find(ts);
  const auto df = feature_descps_.find(ts);
  const auto mf = feature_mask_.find(ts);
  if (kf == feature_kpts_.end() || df == feature_descps_.end() || mf == feature_mask_.end()) {
    return false;
  }
  const int sz_k[3] = {1, rows, 2};
  out.kpts = cv::Mat(3, sz_k, CV_32F, const_cast<uint8_t *>(kf->second.data));
  const int sz_d[3] = {1, rows, static_cast<int>(desc_dim_)};
  out.descps = cv::Mat(3, sz_d, CV_32F, const_cast<uint8_t *>(df->second.data));
  const int sz_m[3] = {1, rows, 1};
  out.mask = cv::Mat(3, sz_m, CV_8U, const_cast<uint8_t *>(mf->second.data));
  return true;
}

cv::Mat MapV3::get_depth(const int64_t ts) const
{
  const auto it = row_of_.find(ts);
  const auto dit = depth_.find(ts);
  if (it == row_of_.end() || dit == depth_.end() || dit->second.shape.size() != 2) {
    return {};
  }
  const int64_t h = dit->second.shape[0], w = dit->second.shape[1];
  if (dit->second.elem_size == 4) {
    // f32 meters: zero-copy view over the sqlite BLOB
    return cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_32F,
      const_cast<uint8_t *>(dit->second.data));
  }
  if (dit->second.elem_size != 2) {
    return {};
  }
  // u16 millimeters: convert to f32 meters on touch (same trade as map_v2).
  const auto * src = reinterpret_cast<const uint16_t *>(dit->second.data);
  cv::Mat out(static_cast<int>(h), static_cast<int>(w), CV_32F);
  const size_t total = static_cast<size_t>(h) * static_cast<size_t>(w);
  for (size_t i = 0; i < total; ++i) {
    out.at<float>(static_cast<int>(i / w), static_cast<int>(i % w)) =
      static_cast<float>(src[i]) * 0.001f;
  }
  return out;
}

}  // namespace tinynav::mapping
