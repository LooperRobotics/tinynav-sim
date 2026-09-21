// Port of reference/tinynav/core/vlad.py — DINOv2 Patch VLAD.
#include "tinynav_cpp/mapping/vlad.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tinynav::mapping {
namespace {

// Port of reference/tinynav/core/vlad.py::_l2_normalize_rows.
Eigen::MatrixXd l2_normalize_rows(const Eigen::MatrixXd& x) {
  Eigen::MatrixXd out(x.rows(), x.cols());
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    const double norm = x.row(i).norm();
    out.row(i) = x.row(i) / std::max(norm, 1e-8);
  }
  return out;
}

// Exact nearest-centre assignment. The Python uses scipy cKDTree (also exact);
// a brute-force scan gives identical labels at these sizes (K ~ 32, C ~ 384).
std::vector<Eigen::Index> nearest_centres(const Eigen::MatrixXd& points,
                                          const Eigen::MatrixXd& centres) {
  std::vector<Eigen::Index> labels(points.rows());
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    (centres.rowwise() - points.row(i)).rowwise().squaredNorm().minCoeff(&labels[i]);
  }
  return labels;
}

}  // namespace

Eigen::MatrixXd train_vocabulary_streaming(
    const TokenIteratorFactory& batch_iterator_factory,
    int vocab_size,
    int epochs,
    int batch_size,
    uint64_t seed) {
  std::mt19937_64 rng(seed);
  Eigen::MatrixXd centres;  // empty until the first batch initialises it
  std::vector<int64_t> counts;

  auto apply_batch = [&](const Eigen::MatrixXd& batch) {
    if (centres.size() == 0) {
      if (batch.rows() < vocab_size) {
        throw std::invalid_argument("Need at least " + std::to_string(vocab_size) +
                                    " tokens in the first batch, got " +
                                    std::to_string(batch.rows()));
      }
      // rng.choice(n, vocab_size, replace=False): a shuffled index subset. Not
      // bit-identical to numpy's PCG64 choice, but seeded and reproducible.
      std::vector<Eigen::Index> order(batch.rows());
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      centres.resize(vocab_size, batch.cols());
      for (int k = 0; k < vocab_size; ++k) {
        centres.row(k) = batch.row(order[k]);
      }
      counts.assign(vocab_size, 0);
    }
    // Assignment against a frozen snapshot of the centres (as the Python does),
    // then a point-by-point decaying running-mean update.
    const Eigen::MatrixXd frozen = centres;
    const std::vector<Eigen::Index> labels = nearest_centres(batch, frozen);
    for (Eigen::Index i = 0; i < batch.rows(); ++i) {
      const Eigen::Index label = labels[i];
      const double eta = 1.0 / static_cast<double>(++counts[label]);
      centres.row(label) = (1.0 - eta) * centres.row(label) + eta * batch.row(i);
    }
  };

  for (int epoch = 0; epoch < epochs; ++epoch) {
    TokenIterator it = batch_iterator_factory();
    Eigen::MatrixXd pending;
    Eigen::MatrixXd frame_tokens;
    while (it(frame_tokens)) {
      const Eigen::MatrixXd normed = l2_normalize_rows(frame_tokens);
      if (pending.rows() == 0) {
        pending = normed;
      } else {
        Eigen::MatrixXd merged(pending.rows() + normed.rows(), pending.cols());
        merged << pending, normed;
        pending = std::move(merged);
      }
      while (pending.rows() >= batch_size) {
        apply_batch(pending.topRows(batch_size));
        pending = pending.bottomRows(pending.rows() - batch_size).eval();
      }
    }
    if (pending.rows() > 0) {
      apply_batch(pending);
    }
  }

  if (centres.size() == 0) {
    throw std::runtime_error("train_vocabulary_streaming: no tokens seen");
  }
  return l2_normalize_rows(centres);
}

Eigen::VectorXd compute_vlad(const Eigen::MatrixXd& patch_tokens,
                             const Eigen::MatrixXd& centres) {
  const Eigen::Index K = centres.rows();
  const Eigen::Index C = centres.cols();
  if (patch_tokens.rows() == 0) {
    return Eigen::VectorXd::Zero(K * C);
  }

  const Eigen::MatrixXd tokens = l2_normalize_rows(patch_tokens);
  const std::vector<Eigen::Index> labels = nearest_centres(tokens, centres);

  Eigen::MatrixXd residuals = Eigen::MatrixXd::Zero(K, C);
  for (Eigen::Index i = 0; i < tokens.rows(); ++i) {
    residuals.row(labels[i]) += tokens.row(i) - centres.row(labels[i]);
  }

  // Intra-normalisation.
  residuals = l2_normalize_rows(residuals);

  // Row-major flatten, matching numpy's C-order reshape(-1).
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rm = residuals;
  Eigen::VectorXd descriptor = Eigen::Map<const Eigen::VectorXd>(rm.data(), K * C);
  const double desc_norm = descriptor.norm();
  if (desc_norm > 1e-8) {
    descriptor /= desc_norm;
  }
  return descriptor;
}

Eigen::MatrixXd compute_vlad_batch(const std::vector<Eigen::MatrixXd>& patch_tokens_list,
                                   const Eigen::MatrixXd& centres) {
  const Eigen::Index K = centres.rows();
  const Eigen::Index C = centres.cols();
  Eigen::MatrixXd descriptors(patch_tokens_list.size(), K * C);
  for (size_t i = 0; i < patch_tokens_list.size(); ++i) {
    descriptors.row(i) = compute_vlad(patch_tokens_list[i], centres).transpose();
  }
  return descriptors;
}

}  // namespace tinynav::mapping
