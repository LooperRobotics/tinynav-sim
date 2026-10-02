// Round-trip tests for the map format v3 (single-file SQLite) loader.
// The fixture is produced from the v2 builder output by the python migrator:
//   /opt/venv/bin/python3 tools/migrate_map_to_v3.py \
//       output/map_build_siglip2/bag_13_43_18 fixtures/map_v3/bag_13_43_18/map.sqlite
// Tests SKIP when either the fixture or the v2 reference npys are absent.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/mapping/map_v3.hpp"

namespace {

const char* kV3Map = "fixtures/map_v3/bag_13_43_18/map.sqlite";
const char* kV2Ref = "output/map_build_siglip2/bag_13_43_18";

bool fixtures_available() {
    return std::filesystem::exists(kV3Map) &&
           std::filesystem::exists(std::string(kV2Ref) + "/pose_timestamps.npy");
}

// Minimal npy reader for the reference arrays: <f8/<f4/<i8/<u2, C-order.
struct NpyData {
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;  // raw little-endian payload

    size_t count() const {
        size_t total = 1;
        for (int64_t dim : shape) total *= static_cast<size_t>(dim);
        return total;
    }
    int elem_size() const {
        return bytes.empty() ? 0 : static_cast<int>(bytes.size() / count());
    }
};

bool load_npy(const std::string& path, NpyData& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[6] = {0};
    in.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return false;
    uint8_t ver[2] = {0, 0};
    in.read(reinterpret_cast<char*>(ver), 2);
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    const auto descr_pos = header.find("'descr'");
    const auto shape_pos = header.find("'shape'");
    if (descr_pos == std::string::npos || shape_pos == std::string::npos) return false;
    int elem = 0;
    if (header.find("<f8", descr_pos) != std::string::npos) elem = 8;
    if (header.find("<f4", descr_pos) != std::string::npos) elem = 4;
    if (header.find("<i8", descr_pos) != std::string::npos) elem = 8;
    if (header.find("u2", descr_pos) != std::string::npos) elem = 2;
    if (elem == 0) return false;
    const auto open = header.find('(', shape_pos);
    const auto close = header.find(')', shape_pos);
    if (open == std::string::npos || close == std::string::npos) return false;
    out.shape.clear();
    std::stringstream ss(header.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) out.shape.push_back(std::stoll(token));
    }
    size_t total = 1;
    for (int64_t dim : out.shape) total *= static_cast<size_t>(dim);
    out.bytes.resize(total * static_cast<size_t>(elem));
    in.read(reinterpret_cast<char*>(out.bytes.data()),
            static_cast<std::streamsize>(out.bytes.size()));
    return static_cast<bool>(in);
}

TEST(map_v3, load_round_trip) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/map_v3 missing; run: /opt/venv/bin/python3 "
                        "tools/migrate_map_to_v3.py output/map_build_siglip2/bag_13_43_18 "
                        "fixtures/map_v3/bag_13_43_18/map.sqlite";
    }
    NpyData ref_ts, ref_pose, ref_vlad, ref_centres, ref_offsets, ref_kpts, ref_depth,
        ref_sem;
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/pose_timestamps.npy", ref_ts));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/pose_matrices.npy", ref_pose));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/vlad_descriptors.npy", ref_vlad));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/vlad_centres.npy", ref_centres));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/feature_offsets.npy", ref_offsets));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/feature_kpts.npy", ref_kpts));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/depth_images.npy", ref_depth));
    ASSERT_TRUE(load_npy(std::string(kV2Ref) + "/semantic_embeddings.npy", ref_sem));

    tinynav::mapping::MapV3 map;
    std::string error;
    ASSERT_TRUE(tinynav::mapping::load_map_v3(kV3Map, map, error)) << error;

    const size_t n = ref_ts.count();
    ASSERT_EQ(map.timestamps.size(), n);
    ASSERT_EQ(map.poses.size(), n);
    const auto* ts = reinterpret_cast<const int64_t*>(ref_ts.bytes.data());
    const double* pose = reinterpret_cast<const double*>(ref_pose.bytes.data());
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(map.timestamps[i], ts[i]);
        const Eigen::Matrix4d& got = map.poses.at(ts[i]);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                EXPECT_NEAR(got(r, c), pose[(i * 4 + static_cast<size_t>(r)) * 4 + c], 1e-9);
            }
        }
    }

    // VLAD index: f32 rows must come back bit-identical.
    ASSERT_EQ(static_cast<size_t>(map.vlad_descriptors.rows()), n);
    ASSERT_EQ(static_cast<size_t>(map.vlad_descriptors.cols()),
              static_cast<size_t>(ref_vlad.shape[1]));
    EXPECT_EQ(std::memcmp(map.vlad_descriptors.data(), ref_vlad.bytes.data(),
                          ref_vlad.bytes.size()),
              0);
    ASSERT_EQ(static_cast<size_t>(map.vlad_centres.rows()),
              static_cast<size_t>(ref_centres.shape[0]));
    ASSERT_EQ(static_cast<size_t>(map.vlad_centres.cols()),
              static_cast<size_t>(ref_centres.shape[1]));
    const float* centres = reinterpret_cast<const float*>(ref_centres.bytes.data());
    for (size_t i = 0; i < ref_centres.count(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(map.vlad_centres.data()[i]), centres[i]);
    }

    // Consumption surface: has_frame / get_features row counts / get_depth size.
    const int64_t* offsets = reinterpret_cast<const int64_t*>(ref_offsets.bytes.data());
    const uint16_t* depth_mm = reinterpret_cast<const uint16_t*>(ref_depth.bytes.data());
    const int h = static_cast<int>(ref_depth.shape[1]);
    const int w = static_cast<int>(ref_depth.shape[2]);
    for (size_t i = 0; i < n; i += 7) {
        const int64_t t = ts[i];
        EXPECT_TRUE(map.has_frame(t));
        tinynav::mapping::MapV2Features feats;
        ASSERT_TRUE(map.get_features(t, feats));
        const int want_rows = static_cast<int>(offsets[i + 1] - offsets[i]);
        EXPECT_EQ(feats.kpts.size[1], want_rows);
        EXPECT_EQ(feats.descps.size[2], 256);
        const cv::Mat depth = map.get_depth(t);
        ASSERT_EQ(depth.rows, h);
        ASSERT_EQ(depth.cols, w);
        if (want_rows > 0) {
            const float ku = feats.kpts.at<float>(0, 0, 0);
            const float kv = feats.kpts.at<float>(0, 0, 1);
            EXPECT_EQ(ku, reinterpret_cast<const float*>(ref_kpts.bytes.data())
                              [(static_cast<size_t>(offsets[i])) * 2]);
            EXPECT_EQ(kv, reinterpret_cast<const float*>(ref_kpts.bytes.data())
                              [(static_cast<size_t>(offsets[i])) * 2 + 1]);
            const int u = std::min(std::max(static_cast<int>(ku), 0), w - 1);
            const int v = std::min(std::max(static_cast<int>(kv), 0), h - 1);
            EXPECT_NEAR(depth.at<float>(v, u),
                        depth_mm[i * static_cast<size_t>(h) * w + static_cast<size_t>(v) * w + u] *
                            0.001f,
                        1e-6f);
        }
    }

    // Semantic block: non-zero row count matches the v2 sidecar.
    const float* sem = reinterpret_cast<const float*>(ref_sem.bytes.data());
    size_t want_sem = 0;
    for (size_t i = 0; i < n; ++i) {
        bool zero = true;
        for (size_t j = 0; j < 768; ++j) {
            if (sem[i * 768 + j] != 0.0f) {
                zero = false;
                break;
            }
        }
        if (!zero) ++want_sem;
    }
    ASSERT_EQ(static_cast<size_t>(map.semantic_embeddings.rows()), n);
    size_t got_sem = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!map.semantic_embeddings.row(static_cast<int>(i)).isZero()) ++got_sem;
    }
    EXPECT_EQ(got_sem, want_sem);
}

TEST(map_v3, missing_file_fails) {
    tinynav::mapping::MapV3 map;
    std::string error;
    EXPECT_FALSE(tinynav::mapping::load_map_v3("fixtures/map_v3/no_such_map.sqlite", map,
                                               error));
    EXPECT_FALSE(error.empty());
}

}  // namespace
