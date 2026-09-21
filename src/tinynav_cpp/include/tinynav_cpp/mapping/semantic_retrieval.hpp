// Port of reference/tinynav/core/semantic_retrieval.py.
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace tinynav::mapping {

// Port of reference/tinynav/core/semantic_retrieval.py::normalize_embedding.
// Throws std::invalid_argument where the Python raises ValueError.
Eigen::VectorXd normalize_embedding(const Eigen::VectorXd& embedding);

// Port of reference/tinynav/core/semantic_retrieval.py::rank_semantic_embeddings.
// `embeddings` is (N, D), one row per timestamp; returns up to top_k
// (timestamp, cosine-score) pairs, best first.
std::vector<std::pair<int64_t, double>> rank_semantic_embeddings(
    const Eigen::VectorXd& query_embedding,
    const Eigen::MatrixXd& embeddings,
    const std::vector<int64_t>& timestamps,
    int top_k = 5);

// Port of reference/tinynav/core/semantic_retrieval.py::load_semantic_embedding_matrix.
// The Python takes a TinyNavDB; here the caller supplies a getter returning the raw
// embedding for a timestamp (existence is the caller's contract, as the Python's
// db.has_semantic_embedding assert was). Returns the (N, D) normalised matrix and
// the parallel timestamp list; an empty input yields a (0, 0) matrix.
std::pair<Eigen::MatrixXd, std::vector<int64_t>> load_semantic_embedding_matrix(
    const std::function<Eigen::VectorXd(int64_t timestamp)>& get_embedding,
    const std::vector<int64_t>& timestamps);

}  // namespace tinynav::mapping
