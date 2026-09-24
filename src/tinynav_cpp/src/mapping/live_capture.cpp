// LiveCapture implementation — see live_capture.hpp for the design contract.
#include "tinynav_cpp/mapping/live_capture.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>

namespace tinynav::mapping
{

namespace
{

void write_small_npy(const std::string & path, const char * descr,
  const std::vector<int64_t> & shape, const void * data, size_t bytes)
{
  std::ofstream out(path, std::ios::binary);
  const std::string header = detail::live_npy_header(descr, shape);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

}  // namespace

LiveCapture::~LiveCapture()
{
  close();
}

bool LiveCapture::ensure_open(std::string & error)
{
  if (open_) {
    return true;
  }
  if (dir_.empty()) {
    error = "live capture dir not set";
    return false;
  }
  // Scratch semantics: the session starts clean (python is_scratch parity).
  std::string cmd = "rm -rf '" + dir_ + "' && mkdir -p '" + dir_ + "'";
  if (std::system(cmd.c_str()) != 0) {
    error = "cannot reset live capture dir " + dir_;
    return false;
  }
  const auto make = [this, &error](std::unique_ptr<GrowFile> & slot,
                     const char * name, const char * descr, size_t dims) {
      slot = std::make_unique<GrowFile>();
      return slot->open(dir_ + "/" + name, descr, dims, error);
    };
  if (!make(depth_file_, "depth_images.npy", "<u2", 3) ||
    !make(kpts_file_, "feature_kpts.npy", "<f4", 2) ||
    !make(descps_file_, "feature_descps.npy", "<f4", 2) ||
    !make(mask_file_, "feature_mask.npy", "|u1", 1) ||
    !make(tokens_file_, "patch_tokens.npy", "<f4", 3))
  {
    return false;
  }
  open_ = true;
  return true;
}

bool LiveCapture::append(int64_t ts, const cv::Mat & depth, const cv::Mat & kpts,
  const cv::Mat & descps, const cv::Mat & mask)
{
  std::string error;
  if (!ensure_open(error)) {
    return false;
  }
  if (depth.type() != CV_32F || depth.empty()) {
    error = "live depth must be non-empty CV_32F";
    return false;
  }
  if (depth_h_ == 0) {
    depth_h_ = depth.rows;
    depth_w_ = depth.cols;
  } else if (depth.rows != depth_h_ || depth.cols != depth_w_) {
    error = "live depth shape changed mid-session";
    return false;
  }

  // f32 meters -> u16 millimeters: 0 stays invalid, negatives clamp to 0,
  // anything past 65.535m clamps (the reloc PnP cutoff is 50m anyway).
  depth_scratch_.resize(static_cast<size_t>(depth_h_) * depth_w_);
  const size_t n_px = depth_scratch_.size();
  for (size_t i = 0; i < n_px; ++i) {
    const float meters = depth.at<float>(
      static_cast<int>(i / depth_w_), static_cast<int>(i % depth_w_));
    const double mm = static_cast<double>(meters) * 1000.0;
    const int v = mm <= 0.0 ? 0 : static_cast<int>(mm + 0.5);
    depth_scratch_[i] = static_cast<uint16_t>(v > 65535 ? 65535 : v);
  }
  if (!depth_file_->append(depth_scratch_.data(), n_px * sizeof(uint16_t), error)) {
    return false;
  }

  // Normalize features to the canonical per-row layout. Empty kpts = a frame
  // with no keypoints: offsets stay put, matching skips it (MapV2 convention).
  size_t n = 0;
  if (!kpts.empty()) {
    n = kpts.total() / 2;
  }
  if (n > 0) {
    if (desc_dim_ == 0) {
      desc_dim_ = static_cast<int64_t>(descps.total()) / static_cast<int64_t>(n);
    }
    if (descps.total() != n * static_cast<size_t>(desc_dim_) ||
      mask.total() != n || kpts.type() != CV_32F || descps.type() != CV_32F)
    {
      error = "live feature shapes inconsistent";
      return false;
    }
    const cv::Mat kpts_c = kpts.isContinuous() ? kpts : kpts.clone();
    const cv::Mat descps_c = descps.isContinuous() ? descps : descps.clone();
    cv::Mat mask_c;
    if (mask.type() == CV_8U) {
      mask_c = mask.isContinuous() ? mask : mask.clone();
    } else {
      mask.convertTo(mask_c, CV_8U);
    }
    if (!kpts_file_->append(kpts_c.data, n * 2 * sizeof(float), error) ||
      !descps_file_->append(descps_c.data,
        n * static_cast<size_t>(desc_dim_) * sizeof(float), error) ||
      !mask_file_->append(mask_c.data, n, error))
    {
      return false;
    }
  }

  if (feature_offsets_.empty()) {
    feature_offsets_.push_back(0);  // the v2 leading zero
  }
  feature_offsets_.push_back(feature_offsets_.back() + static_cast<int64_t>(n));
  timestamps_.push_back(ts);
  row_of_[ts] = static_cast<uint32_t>(timestamps_.size() - 1);
  return true;
}

bool LiveCapture::get_features(int64_t ts, MapV2Features & out) const
{
  out = MapV2Features{};
  const auto it = row_of_.find(ts);
  if (it == row_of_.end() || !open_) {
    return false;
  }
  const size_t i = it->second;
  // feature_offsets_ is cumulative with the v2 leading 0: [0, n0, n0+n1, ...]
  const int64_t begin = feature_offsets_[i];
  const int rows = static_cast<int>(feature_offsets_[i + 1] - begin);
  if (rows <= 0) {
    return true;
  }
  const int sz_k[3] = {1, rows, 2};
  out.kpts = cv::Mat(3, sz_k, CV_32F,
    const_cast<void *>(kpts_file_->view(static_cast<size_t>(begin) * 2 * sizeof(float))));
  const int sz_d[3] = {1, rows, static_cast<int>(desc_dim_)};
  out.descps = cv::Mat(3, sz_d, CV_32F,
    const_cast<void *>(descps_file_->view(
      static_cast<size_t>(begin) * static_cast<size_t>(desc_dim_) * sizeof(float))));
  const int sz_m[3] = {1, rows, 1};
  out.mask = cv::Mat(3, sz_m, CV_8U,
    const_cast<void *>(mask_file_->view(static_cast<size_t>(begin))));
  return true;
}

cv::Mat LiveCapture::get_depth(int64_t ts) const
{
  const auto it = row_of_.find(ts);
  if (it == row_of_.end() || !open_ || depth_h_ == 0) {
    return {};
  }
  const auto * src = static_cast<const uint16_t *>(
    depth_file_->view(static_cast<size_t>(it->second) *
                      static_cast<size_t>(depth_h_) * depth_w_ * sizeof(uint16_t)));
  cv::Mat out(depth_h_, depth_w_, CV_32F);
  const size_t n = static_cast<size_t>(depth_h_) * depth_w_;
  for (size_t i = 0; i < n; ++i) {
    out.at<float>(static_cast<int>(i / depth_w_), static_cast<int>(i % depth_w_)) =
      static_cast<float>(src[i]) * 0.001f;
  }
  return out;
}

bool LiveCapture::append_tokens(int64_t ts, const cv::Mat & tokens)
{
  std::string error;
  if (!open_ || closed_) {
    return false;
  }
  // The token row hangs off an already-captured frame, once.
  if (row_of_.count(ts) == 0) {
    return false;
  }
  if (tok_row_of_.count(ts) != 0) {
    return true;
  }
  if (tokens.empty() || tokens.type() != CV_32F) {
    return false;
  }
  if (tok_rows_ == 0) {
    tok_rows_ = tokens.rows;
    tok_cols_ = tokens.cols;
  } else if (tokens.rows != tok_rows_ || tokens.cols != tok_cols_) {
    return false;
  }
  const cv::Mat c = tokens.isContinuous() ? tokens : tokens.clone();
  if (!tokens_file_->append(
      c.data, static_cast<size_t>(tok_rows_) * tok_cols_ * sizeof(float), error))
  {
    return false;
  }
  tok_row_of_[ts] = static_cast<uint32_t>(tok_count_++);
  return true;
}

bool LiveCapture::get_tokens(int64_t ts, Eigen::MatrixXd & out) const
{
  const auto it = tok_row_of_.find(ts);
  if (it == tok_row_of_.end() || !open_ || tok_rows_ == 0 || tok_cols_ == 0) {
    return false;
  }
  const size_t n = static_cast<size_t>(tok_rows_) * tok_cols_;
  const auto * src = static_cast<const float *>(
    tokens_file_->view(it->second * n * sizeof(float)));
  out = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::RowMajor>>(src, static_cast<Eigen::Index>(tok_rows_),
                      static_cast<Eigen::Index>(tok_cols_)).cast<double>();
  return true;
}

void LiveCapture::reset()
{
  close();
  closed_ = false;
  open_ = false;
  timestamps_.clear();
  feature_offsets_.clear();
  row_of_.clear();
  tok_row_of_.clear();
  tok_count_ = 0;
  depth_h_ = depth_w_ = 0;
  desc_dim_ = 0;
  tok_rows_ = tok_cols_ = 0;
  depth_scratch_.clear();
  mask_scratch_.clear();
}

void LiveCapture::close()
{
  if (!open_ || closed_) {
    return;
  }
  closed_ = true;
  open_ = false;
  std::string error;

  // Patch the streaming files into valid npys with their final shapes.
  depth_file_->finalize(
    {static_cast<int64_t>(timestamps_.size()), depth_h_, depth_w_}, error);
  const int64_t total_rows = feature_offsets_.empty() ? 0 : feature_offsets_.back();
  kpts_file_->finalize({total_rows, 2}, error);
  descps_file_->finalize({total_rows, desc_dim_}, error);
  mask_file_->finalize({total_rows}, error);
  tokens_file_->finalize({static_cast<int64_t>(tok_count_), tok_rows_, tok_cols_},
    error);

  // feature_offsets_ already carries the v2 leading zero
  write_small_npy(dir_ + "/feature_offsets.npy", "<i8",
    {static_cast<int64_t>(feature_offsets_.size())}, feature_offsets_.data(),
    feature_offsets_.size() * sizeof(int64_t));
  write_small_npy(dir_ + "/timestamps.npy", "<i8", {timestamps_.size()},
    timestamps_.data(), timestamps_.size() * sizeof(int64_t));

  std::ofstream meta(dir_ + "/meta.json");
  meta << "{\"format\": \"tinynav_live_v1\", \"depth\": \"u16_mm\", "
       << "\"keyframes\": " << timestamps_.size() << ", \"h\": " << depth_h_
       << ", \"w\": " << depth_w_ << ", \"desc_dim\": " << desc_dim_
       << ", \"patch_frames\": " << tok_count_
       << ", \"patch_rows\": " << tok_rows_
       << ", \"patch_cols\": " << tok_cols_ << "}";
}

}  // namespace tinynav::mapping
