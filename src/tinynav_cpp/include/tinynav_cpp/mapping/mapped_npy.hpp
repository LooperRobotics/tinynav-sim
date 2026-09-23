#ifndef TINYNAV_CPP__MAPPING__MAPPED_NPY_HPP_
#define TINYNAV_CPP__MAPPING__MAPPED_NPY_HPP_

#include <cstdint>
#include <string>
#include <sstream>
#include <algorithm>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tinynav::mapping
{

// Read-only mmap view of one .npy array. Python's TinyNavDB keeps the big
// per-keyframe arrays in a shelve (disk-backed dict) and fetches rows only
// when a reloc candidate needs them; this is the map-format-v2 counterpart:
// the flat npy stays mapped and a row is touched only on access, so RSS no
// longer scales with keyframe count. C-order <f4/<i8/u1 only — the
// exporter's contract (tools/export_map_v2.py). Move-only RAII.
class MappedNpy
{
public:
  enum class DType { F4, F8, I8, U1, U2 };

  MappedNpy() = default;
  MappedNpy(MappedNpy && other) noexcept { steal(other); }
  MappedNpy & operator=(MappedNpy && other) noexcept
  {
    if (this != &other) {
      release();
      steal(other);
    }
    return *this;
  }
  MappedNpy(const MappedNpy &) = delete;
  MappedNpy & operator=(const MappedNpy &) = delete;
  ~MappedNpy() { release(); }

  bool open(const std::string & path, std::string & error)
  {
    release();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      error = "cannot open " + path;
      return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
      error = "cannot stat " + path;
      ::close(fd);
      return false;
    }
    const size_t file_size = static_cast<size_t>(st.st_size);

    char magic[6] = {0};
    if (::pread(fd, magic, 6, 0) != 6 || std::string(magic, 6) != "\x93NUMPY") {
      error = "not an npy file: " + path;
      ::close(fd);
      return false;
    }
    uint8_t ver[2] = {0, 0};
    if (::pread(fd, ver, 2, 6) != 2) {
      error = "truncated npy header: " + path;
      ::close(fd);
      return false;
    }
    uint64_t header_len = 0;
    size_t prefix = 0;
    if (ver[0] == 1) {
      uint16_t len16 = 0;
      if (::pread(fd, &len16, 2, 8) != 2) {
        error = "truncated npy header: " + path;
        ::close(fd);
        return false;
      }
      header_len = len16;
      prefix = 10;
    } else if (ver[0] == 2 || ver[0] == 3) {
      uint32_t len32 = 0;
      if (::pread(fd, &len32, 4, 8) != 4) {
        error = "truncated npy header: " + path;
        ::close(fd);
        return false;
      }
      header_len = len32;
      prefix = 12;
    } else {
      error = "unsupported npy version in " + path;
      ::close(fd);
      return false;
    }
    std::string header(header_len, '\0');
    if (::pread(fd, header.data(), header_len, static_cast<off_t>(prefix)) !=
      static_cast<ssize_t>(header_len))
    {
      error = "truncated npy header: " + path;
      ::close(fd);
      return false;
    }

    const auto descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) {
      error = "npy header has no descr: " + path;
      ::close(fd);
      return false;
    }
    if (header.find("<f4", descr_pos) != std::string::npos) {
      dtype_ = DType::F4;
    } else if (header.find("<f8", descr_pos) != std::string::npos) {
      dtype_ = DType::F8;
    } else if (header.find("<i8", descr_pos) != std::string::npos) {
      dtype_ = DType::I8;
    } else if (header.find("<u2", descr_pos) != std::string::npos) {
      dtype_ = DType::U2;
    } else if (header.find("u1", descr_pos) != std::string::npos) {
      dtype_ = DType::U1;
    } else {
      error = "unsupported dtype in " + path + " (want <f4/<f8/<i8/<u2/u1)";
      ::close(fd);
      return false;
    }
    if (header.find("True", header.find("'fortran_order'")) != std::string::npos) {
      error = "fortran-order npy not supported: " + path;
      ::close(fd);
      return false;
    }
    const auto shape_pos = header.find("'shape'");
    const auto open_pos = header.find('(', shape_pos);
    const auto close_pos = header.find(')', shape_pos);
    if (shape_pos == std::string::npos || open_pos == std::string::npos ||
      close_pos == std::string::npos)
    {
      error = "npy header has no shape: " + path;
      ::close(fd);
      return false;
    }
    shape_.clear();
    std::stringstream ss(header.substr(open_pos + 1, close_pos - open_pos - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
      token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
      if (!token.empty()) {
        shape_.push_back(std::stoll(token));
      }
    }
    if (shape_.empty()) {
      error = "npy holds no rows: " + path;
      ::close(fd);
      return false;
    }

    size_t elem = 0;
    switch (dtype_) {
      case DType::F4: elem = 4; break;
      case DType::F8: elem = 8; break;
      case DType::I8: elem = 8; break;
      case DType::U1: elem = 1; break;
      case DType::U2: elem = 2; break;
    }
    rows_ = shape_[0];
    size_t row_elems = 1;
    for (size_t i = 1; i < shape_.size(); ++i) {
      row_elems *= static_cast<size_t>(shape_[i]);
    }
    row_bytes_ = row_elems * elem;
    data_offset_ = prefix + header_len;
    if (data_offset_ + rows_ * row_bytes_ > file_size) {
      error = "truncated npy data: " + path;
      ::close(fd);
      return false;
    }

    void * base =
      ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
      error = "mmap failed: " + path;
      return false;
    }
    // Reloc reads a few candidate frames per query, scattered across the file —
    // readahead of whole neighbours would page in data nobody asked for.
    ::madvise(base, file_size, MADV_RANDOM);
    base_ = static_cast<char *>(base);
    file_size_ = file_size;
    path_ = path;
    return true;
  }

  bool ok() const { return base_ != nullptr; }
  DType dtype() const { return dtype_; }
  const std::vector<int64_t> & shape() const { return shape_; }
  int64_t rows() const { return rows_; }
  size_t row_bytes() const { return row_bytes_; }
  size_t file_bytes() const { return file_size_; }

  // Pointer to row i. The mapping is read-only; the const_cast is for
  // cv::Mat's non-const constructor — writing through it would SIGSEGV.
  const void * row(size_t i) const
  {
    return base_ + data_offset_ + i * row_bytes_;
  }

private:
  void steal(MappedNpy & other)
  {
    base_ = other.base_;
    file_size_ = other.file_size_;
    data_offset_ = other.data_offset_;
    rows_ = other.rows_;
    row_bytes_ = other.row_bytes_;
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    path_ = std::move(other.path_);
    other.base_ = nullptr;
  }
  void release()
  {
    if (base_ != nullptr) {
      ::munmap(base_, file_size_);
      base_ = nullptr;
    }
  }

  char * base_ = nullptr;
  size_t file_size_ = 0;
  size_t data_offset_ = 0;
  int64_t rows_ = 0;
  size_t row_bytes_ = 0;
  std::vector<int64_t> shape_;
  DType dtype_ = DType::F4;
  std::string path_;
};

}  // namespace tinynav::mapping

#endif  // TINYNAV_CPP__MAPPING__MAPPED_NPY_HPP_
