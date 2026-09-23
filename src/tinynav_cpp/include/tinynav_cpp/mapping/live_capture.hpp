#ifndef TINYNAV_CPP__MAPPING__LIVE_CAPTURE_HPP_
#define TINYNAV_CPP__MAPPING__LIVE_CAPTURE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core.hpp>

#include "tinynav_cpp/mapping/map_v2.hpp"

namespace tinynav::mapping
{

// Fixed live npy header block: reserved at open with a zeroed shape, patched
// in place at close with the final shape — data offset never moves.
inline constexpr size_t kLiveNpyHeaderLen = 128;

namespace detail
{
// Full 128-byte npy v1 prefix: magic + version + header_len + dict, padded so
// finalize() can overwrite it byte-for-byte without moving the data offset.
inline std::string live_npy_header(const char * descr,
  const std::vector<int64_t> & shape)
{
  std::string dims;
  for (size_t i = 0; i < shape.size(); ++i) {
    dims += (i ? "," : "") + std::to_string(shape[i]);
  }
  // the template's ",)" supplies the trailing comma numpy expects
  std::string dict =
    std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" +
    dims + ",), }";
  dict.resize(kLiveNpyHeaderLen - 10 - 1, ' ');
  dict.push_back('\n');
  std::string header(10, '\0');
  header[0] = '\x93';
  header[1] = 'N';
  header[2] = 'U';
  header[3] = 'M';
  header[4] = 'P';
  header[5] = 'Y';
  header[6] = 1;  // npy version 1.0
  header[7] = 0;
  const auto len = static_cast<uint16_t>(dict.size());
  std::memcpy(header.data() + 8, &len, 2);
  header += dict;
  return header;  // exactly kLiveNpyHeaderLen bytes
}
}  // namespace detail

// Disk-backed scratch store for the LIVE keyframes of one navigation session —
// the C++ counterpart of map_node.py's nav_temp_db (a scratch TinyNavDB).
// Depth planes and SuperPoint features are appended to a columnar on-disk
// layout and read back per loop-closure candidate, so component RSS no longer
// scales with session length (the in-memory maps it replaces grew ~1.9MB per
// keyframe, unbounded). Embeddings stay in RAM — find_loop scans them on
// every keyframe — and poses never live here (the pose graph owns them).
//
// Layout = the map-format-v2 file set, written incrementally: depth_images
// (<u2 millimeters — half the python f32 scratch footprint), feature_{kpts,
// descps,mask} + feature_offsets, timestamps + meta.json at close(). After
// close() the directory plus poses and a VLAD index is a valid v2 map subset
// — online capture and offline rebuild share one artifact, which is why the
// regression test loads it through load_map_v2.
//
// Scratch semantics (python is_scratch parity): the first append wipes the
// directory; a crash loses the session — the accepted nav_temp contract.
// No internal locking: one callback thread owns it, like every component
// store. Depth is CV_32F meters at the edges, u16 mm on disk (1mm
// quantization is orders below stereo noise); features are non-owning mmap
// views, depth comes back as an owning CV_32F Mat after conversion.
class LiveCapture
{
public:
  LiveCapture() = default;
  ~LiveCapture();
  LiveCapture(const LiveCapture &) = delete;
  LiveCapture & operator=(const LiveCapture &) = delete;

  // Where the session lives; only read at the first append.
  void set_dir(const std::string & dir) { dir_ = dir; }

  // depth: CV_32F HxW meters. kpts: [.,N,2] CV_32F; descps: [.,N,D] CV_32F;
  // mask: CV_8U (CV_32F accepted, converted). First call pins H/W/D and wipes
  // the directory; later shape mismatches fail. Returns false on IO error —
  // the caller should treat the frame as outside loop closure (and not index
  // its embedding), navigation is unaffected.
  bool append(int64_t ts, const cv::Mat & depth, const cv::Mat & kpts,
    const cv::Mat & descps, const cv::Mat & mask);

  bool has(int64_t ts) const { return row_of_.count(ts) != 0; }
  size_t size() const { return timestamps_.size(); }

  // Owning CV_32F HxW meters (converted from the u16 mm row); empty when the
  // frame is unknown or nothing was appended yet.
  cv::Mat get_depth(int64_t ts) const;

  // Non-owning views into the mapping; the LiveCapture owner must outlive
  // them (true at the only call site, loop-closure scope). rows==0 keyframes
  // yield empty Mats, same convention as MapV2::get_features.
  bool get_features(int64_t ts, MapV2Features & out) const;

  // Patches the npy headers with the final shapes and writes
  // feature_offsets.npy / timestamps.npy / meta.json. Best-effort, idempotent;
  // the destructor calls it.
  void close();

  // Append-only file with a lazy read mapping: writes go through pwrite (the
  // page cache is the source of truth); the read mapping is grown ahead of
  // the tail on every append, so views never fault past the mapped window
  // and reads are zero-copy.
  class GrowFile
  {
  public:
    GrowFile() = default;
    ~GrowFile() { release(); }
    GrowFile(const GrowFile &) = delete;
    GrowFile & operator=(const GrowFile &) = delete;

    bool open(const std::string & path, const char * descr, size_t dims,
      std::string & error)
    {
      path_ = path;
      descr_ = descr;
      dims_ = dims;
      fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
      if (fd_ < 0) {
        error = "cannot create " + path;
        return false;
      }
      const std::string header = detail::live_npy_header(descr,
          std::vector<int64_t>(dims, 0));
      if (::write(fd_, header.data(), header.size()) !=
        static_cast<ssize_t>(header.size()))
      {
        error = "cannot write header: " + path;
        return false;
      }
      tail_ = header.size();
      data_offset_ = tail_;
      return true;
    }

    bool append(const void * data, size_t bytes, std::string & error)
    {
      if (::pwrite(fd_, data, bytes, static_cast<off_t>(tail_)) !=
        static_cast<ssize_t>(bytes))
      {
        error = "append failed: " + path_;
        return false;
      }
      tail_ += bytes;
      return ensure_mapped(tail_);
    }

    // Read view at a data offset; valid because append() always maps the tail.
    const void * view(size_t data_offset) const
    {
      return map_ + data_offset_ + data_offset;
    }

    // Patch the placeholder header with the final shape and drop the mapping.
    bool finalize(const std::vector<int64_t> & shape, std::string & error)
    {
      if (fd_ < 0) {
        return true;
      }
      const std::string header = detail::live_npy_header(descr_, shape);
      if (::pwrite(fd_, header.data(), header.size(), 0) !=
        static_cast<ssize_t>(header.size()))
      {
        error = "cannot finalize header: " + path_;
        return false;
      }
      if (::ftruncate(fd_, static_cast<off_t>(tail_)) != 0) {
        error = "cannot truncate: " + path_;
        return false;
      }
      release();
      return true;
    }

  private:
    bool ensure_mapped(size_t upto)
    {
      if (upto <= map_cap_) {
        return true;
      }
      size_t cap = map_cap_ ? map_cap_ : 64UL * 1024 * 1024;
      while (cap < upto) {
        cap *= 2;
      }
      void * next = ::mmap(nullptr, cap, PROT_READ, MAP_SHARED, fd_, 0);
      if (next == MAP_FAILED) {
        // The data is safe in the page cache either way; only reads suffer.
        return true;
      }
      if (map_ != nullptr) {
        ::munmap(map_, map_cap_);
      }
      map_ = static_cast<char *>(next);
      map_cap_ = cap;
      return true;
    }

    void release()
    {
      if (map_ != nullptr) {
        ::munmap(map_, map_cap_);
        map_ = nullptr;
        map_cap_ = 0;
      }
      if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
      }
    }

    int fd_ = -1;
    std::string path_;
    const char * descr_ = "";
    size_t dims_ = 0;
    size_t tail_ = 0;        // absolute file size (header included)
    size_t data_offset_ = 0; // where row data starts
    char * map_ = nullptr;
    size_t map_cap_ = 0;
  };

private:
  bool ensure_open(std::string & error);

  std::string dir_;
  bool open_ = false;
  bool closed_ = false;

  std::unique_ptr<GrowFile> depth_file_;   // depth_images <u2 mm [N,H,W]
  std::unique_ptr<GrowFile> kpts_file_;    // feature_kpts <f4 [M,2]
  std::unique_ptr<GrowFile> descps_file_;  // feature_descps <f4 [M,D]
  std::unique_ptr<GrowFile> mask_file_;    // feature_mask u1 [M]

  std::vector<int64_t> timestamps_;
  std::vector<int64_t> feature_offsets_;  // cumulative kpt rows, leading 0
  std::unordered_map<int64_t, uint32_t> row_of_;

  int depth_h_ = 0, depth_w_ = 0;
  int64_t desc_dim_ = 0;
  std::vector<uint16_t> depth_scratch_;   // reusable f32m -> u16mm buffer
  std::vector<uint8_t> mask_scratch_;
};

}  // namespace tinynav::mapping

#endif  // TINYNAV_CPP__MAPPING__LIVE_CAPTURE_HPP_
