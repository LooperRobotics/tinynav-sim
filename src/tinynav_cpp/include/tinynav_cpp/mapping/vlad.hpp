// Port of reference/tinynav/core/vlad.py — DINOv2 Patch VLAD.
// numpy float32 arrays become Eigen double matrices (repo rule: double everywhere).
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace tinynav::mapping {

// Pull-iterator over per-keyframe (N_i, C) patch-token matrices: returns false when
// exhausted. The factory is called once per epoch and must return a fresh iterator
// (mirrors the Python `batch_iterator_factory` zero-arg callable).
using TokenIterator = std::function<bool(Eigen::MatrixXd& out_tokens)>;
using TokenIteratorFactory = std::function<TokenIterator()>;

// Port of reference/tinynav/core/vlad.py::train_vocabulary_streaming.
// Online (Robbins-Monro) k-means over streamed patch tokens. Throws
// std::invalid_argument if the first batch holds fewer than vocab_size tokens,
// std::runtime_error if no tokens are seen at all (the Python returns None and
// crashes downstream; failing loudly here keeps the error at its source).
Eigen::MatrixXd train_vocabulary_streaming(
    const TokenIteratorFactory& batch_iterator_factory,
    int vocab_size = 32,
    int epochs = 5,
    int batch_size = 1024,
    uint64_t seed = 42);

// Port of reference/tinynav/core/vlad.py::compute_vlad.
// Returns the (K*C,) L2-normalised VLAD descriptor (row-major flatten of the
// (K, C) residuals, matching numpy's C-order reshape).
Eigen::VectorXd compute_vlad(const Eigen::MatrixXd& patch_tokens,
                             const Eigen::MatrixXd& centres);

// Port of reference/tinynav/core/vlad.py::compute_vlad_batch.
// Returns (list.size(), K*C).
Eigen::MatrixXd compute_vlad_batch(const std::vector<Eigen::MatrixXd>& patch_tokens_list,
                                   const Eigen::MatrixXd& centres);

}  // namespace tinynav::mapping
