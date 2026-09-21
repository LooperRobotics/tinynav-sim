// Port of reference/tinynav/core/semantic_retrieval.py.
#include "tinynav_cpp/mapping/semantic_retrieval.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tinynav::mapping {

Eigen::VectorXd normalize_embedding(const Eigen::VectorXd& embedding) {
  const double norm = embedding.norm();
  if (!std::isfinite(norm) || norm <= 0.0) {
    throw std::invalid_argument("embedding norm must be finite and positive");
  }
  return embedding / norm;
}

std::vector<std::pair<int64_t, double>> rank_semantic_embeddings(
    const Eigen::VectorXd& query_embedding,
    const Eigen::MatrixXd& embeddings,
    const std::vector<int64_t>& timestamps,
    int top_k) {
  if (top_k <= 0) {
    throw std::invalid_argument("top_k must be positive");
  }
  const Eigen::VectorXd query = normalize_embedding(query_embedding);
  if (embeddings.rows() != static_cast<Eigen::Index>(timestamps.size()) ||
      embeddings.cols() != query.size()) {
    throw std::invalid_argument("embeddings shape must be (len(timestamps), len(query))");
  }

  const Eigen::VectorXd scores = embeddings * query;
  const Eigen::Index n = scores.size();
  std::vector<Eigen::Index> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](Eigen::Index a, Eigen::Index b) { return scores[a] < scores[b]; });

  // np.argsort(scores)[-top_k:][::-1] — the top-k scores, best first.
  std::vector<std::pair<int64_t, double>> out;
  const Eigen::Index start = std::max<Eigen::Index>(0, n - top_k);
  for (Eigen::Index i = n - 1; i >= start; --i) {
    out.emplace_back(timestamps[order[i]], scores[order[i]]);
  }
  return out;
}

std::pair<Eigen::MatrixXd, std::vector<int64_t>> load_semantic_embedding_matrix(
    const std::function<Eigen::VectorXd(int64_t timestamp)>& get_embedding,
    const std::vector<int64_t>& timestamps) {
  if (timestamps.empty()) {
    return {Eigen::MatrixXd(0, 0), {}};
  }
  std::vector<Eigen::VectorXd> embeddings;
  embeddings.reserve(timestamps.size());
  for (const int64_t timestamp : timestamps) {
    embeddings.push_back(normalize_embedding(get_embedding(timestamp)));
  }
  Eigen::MatrixXd matrix(embeddings.size(), embeddings.front().size());
  for (size_t i = 0; i < embeddings.size(); ++i) {
    if (embeddings[i].size() != matrix.cols()) {
      throw std::invalid_argument("embedding dimensions differ across timestamps");
    }
    matrix.row(i) = embeddings[i].transpose();
  }
  return {matrix, timestamps};
}

}  // namespace tinynav::mapping
