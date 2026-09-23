// Alignment tests for the map format v2 loader and the reloc unprojection.
// Fixtures are synthetic (tools/export_map_v2_fixtures.py): a tiny Python-format
// map (pickled poses + shelve dbs) exported by tools/export_map_v2.py, plus the
// expected arrays. Tests SKIP when fixtures/ is absent.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/mapping/map_v2.hpp"

namespace {

bool fixtures_available() {
    return std::filesystem::exists("fixtures/map_v2/v2/pose_timestamps.npy");
}

// i64 npy reader (timestamps; exact — doubles would round 1.7e18-scale ns).
bool load_fixture_npy_i64(const std::string& path, std::vector<int64_t>& data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[6] = {0};
    in.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return false;
    uint8_t ver[2] = {0, 0};
    in.read(reinterpret_cast<char*>(ver), 2);
    uint16_t header_len = 0;
    if (ver[0] == 1) {
        in.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t len32 = 0;
        in.read(reinterpret_cast<char*>(&len32), 4);
        header_len = static_cast<uint16_t>(len32);
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (header.find("<i8") == std::string::npos) return false;
    const auto shape_pos = header.find("'shape'");
    const auto open = header.find('(', shape_pos);
    const auto close = header.find(')', shape_pos);
    std::stringstream ss(header.substr(open + 1, close - open - 1));
    std::string token;
    size_t total = 1;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) total *= static_cast<size_t>(std::stoll(token));
    }
    data.resize(total);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(int64_t)));
    return static_cast<bool>(in);
}

bool load_fixture_npy_doubles(const std::string& path, std::vector<int64_t>& shape,
                              std::vector<double>& data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[6] = {0};
    in.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return false;
    uint8_t ver[2] = {0, 0};
    in.read(reinterpret_cast<char*>(ver), 2);
    uint16_t header_len = 0;
    if (ver[0] == 1) {
        in.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t len32 = 0;
        in.read(reinterpret_cast<char*>(&len32), 4);
        header_len = static_cast<uint16_t>(len32);
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    const auto descr_pos = header.find("'descr'");
    const auto shape_pos = header.find("'shape'");
    if (descr_pos == std::string::npos || shape_pos == std::string::npos) return false;
    const bool is_f32 = header.find("<f4", descr_pos) != std::string::npos;
    const bool is_f64 = header.find("<f8", descr_pos) != std::string::npos;
    if (!is_f32 && !is_f64) return false;
    const auto open = header.find('(', shape_pos);
    const auto close = header.find(')', shape_pos);
    std::stringstream ss(header.substr(open + 1, close - open - 1));
    std::string token;
    shape.clear();
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) shape.push_back(std::stoll(token));
    }
    size_t total = 1;
    for (int64_t dim : shape) total *= static_cast<size_t>(dim);
    data.resize(total);
    if (is_f32) {
        std::vector<float> raw(total);
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total * sizeof(float)));
        for (size_t i = 0; i < total; ++i) data[i] = raw[i];
    } else {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(double)));
    }
    return static_cast<bool>(in);
}

}  // namespace

TEST(map_v2_alignment, load_round_trip) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/map_v2 not exported; run tools/export_map_v2_fixtures.py";
    }
    tinynav::mapping::MapV2 map;
    std::string error;
    ASSERT_TRUE(tinynav::mapping::load_map_v2("fixtures/map_v2/v2", map, error)) << error;

    std::vector<int64_t> shape;
    std::vector<double> data;
    const auto expect_shape = [&](const std::vector<int64_t>& want) {
        ASSERT_EQ(shape.size(), want.size());
        for (size_t i = 0; i < want.size(); ++i) ASSERT_EQ(shape[i], want[i]);
    };

    // timestamps + poses
    std::vector<int64_t> expected_timestamps;
    ASSERT_TRUE(load_fixture_npy_i64("fixtures/map_v2/expected/pose_timestamps.npy", expected_timestamps));
    ASSERT_EQ(map.timestamps.size(), expected_timestamps.size());
    for (size_t i = 0; i < map.timestamps.size(); ++i) {
        ASSERT_EQ(map.timestamps[i], expected_timestamps[i]);
    }
    ASSERT_TRUE(load_fixture_npy_doubles("fixtures/map_v2/expected/pose_matrices.npy", shape, data));
    expect_shape({5, 4, 4});
    for (size_t i = 0; i < map.timestamps.size(); ++i) {
        const Eigen::Matrix4d& pose = map.poses.at(map.timestamps[i]);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                EXPECT_NEAR(pose(r, c), data[i * 16 + r * 4 + c], 1e-9);
            }
        }
    }

    // VLAD index
    ASSERT_TRUE(load_fixture_npy_doubles("fixtures/map_v2/expected/vlad_centres.npy", shape, data));
    expect_shape({4, 8});
    ASSERT_EQ(map.vlad_centres.rows(), 4);
    ASSERT_EQ(map.vlad_centres.cols(), 8);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 8; ++c) {
            EXPECT_NEAR(map.vlad_centres(r, c), data[r * 8 + c], 1e-6);
        }
    }
    ASSERT_TRUE(load_fixture_npy_doubles("fixtures/map_v2/expected/vlad_descriptors.npy", shape, data));
    expect_shape({5, 8});
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 8; ++c) {
            EXPECT_NEAR(map.vlad_descriptors(r, c), data[r * 8 + c], 1e-6);
        }
    }

    // features: canonical packing, rows ordered by timestamp
    ASSERT_TRUE(load_fixture_npy_doubles("fixtures/map_v2/expected/feature_kpts.npy", shape, data));
    ASSERT_EQ(shape.size(), 2);
    ASSERT_EQ(shape[1], 2);
    const int m = static_cast<int>(shape[0]);
    std::vector<int64_t> offsets;
    ASSERT_TRUE(load_fixture_npy_i64("fixtures/map_v2/expected/feature_offsets.npy", offsets));
    ASSERT_EQ(offsets.size(), 6u);
    for (int i = 0; i < 5; ++i) {
        tinynav::mapping::MapV2Features feat;
        ASSERT_TRUE(map.get_features(map.timestamps[static_cast<size_t>(i)], feat));
        const int rows = static_cast<int>(offsets[i + 1] - offsets[i]);
        ASSERT_EQ(feat.kpts.dims, 3);
        ASSERT_EQ(feat.kpts.size[0], 1);
        ASSERT_EQ(feat.kpts.size[1], rows);
        ASSERT_EQ(feat.kpts.size[2], 2);
        ASSERT_EQ(feat.descps.dims, 3);
        ASSERT_EQ(feat.descps.size[1], rows);
        ASSERT_EQ(feat.descps.size[2], 256);
        ASSERT_EQ(feat.mask.dims, 3);
        ASSERT_EQ(feat.mask.size[1], rows);
        for (int j = 0; j < rows; ++j) {
            const int src = offsets[i] + j;
            EXPECT_FLOAT_EQ(feat.kpts.at<float>(0, j, 0),
                            static_cast<float>(data[src * 2]));
            EXPECT_FLOAT_EQ(feat.kpts.at<float>(0, j, 1),
                            static_cast<float>(data[src * 2 + 1]));
        }
    }
    // descps + mask byte-checked against the exporter's own output (same values
    // the fixture generator seeded).
    ASSERT_TRUE(load_fixture_npy_doubles("fixtures/map_v2/expected/feature_descps.npy", shape, data));
    for (int i = 0; i < 5; ++i) {
        tinynav::mapping::MapV2Features feat;
        ASSERT_TRUE(map.get_features(map.timestamps[static_cast<size_t>(i)], feat));
        const int rows = static_cast<int>(offsets[i + 1] - offsets[i]);
        for (int j = 0; j < rows; ++j) {
            const int src = offsets[i] + j;
            for (int d = 0; d < 256; ++d) {
                EXPECT_FLOAT_EQ(feat.descps.at<float>(0, j, d),
                                static_cast<float>(data[src * 256 + d]));
            }
        }
    }

    // depth planes (u16 mm on disk; the reader converts back to f32 meters —
    // allow the ±0.5mm quantization)
    ASSERT_TRUE(load_fixture_npy_doubles("fixtures/map_v2/expected/depth_images.npy", shape, data));
    expect_shape({5, 12, 16});
    for (int i = 0; i < 5; ++i) {
        const cv::Mat depth = map.get_depth(map.timestamps[static_cast<size_t>(i)]);
        ASSERT_FALSE(depth.empty());
        ASSERT_EQ(depth.rows, 12);
        ASSERT_EQ(depth.cols, 16);
        ASSERT_EQ(depth.type(), CV_32F);
        for (int r = 0; r < 12; ++r) {
            for (int c = 0; c < 16; ++c) {
                EXPECT_NEAR(depth.at<float>(r, c),
                            static_cast<float>(data[i * 12 * 16 + r * 16 + c]),
                            5.1e-4);
            }
        }
    }
}

TEST(map_v2_alignment, keypoint_with_depth_to_3d) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/map_v2 not exported; run tools/export_map_v2_fixtures.py";
    }
    tinynav::mapping::MapV2 map;
    std::string error;
    ASSERT_TRUE(tinynav::mapping::load_map_v2("fixtures/map_v2/v2", map, error)) << error;

    const Eigen::Matrix4d& pose = map.poses.at(map.timestamps[0]);
    const cv::Mat depth = map.get_depth(map.timestamps[0]);
    ASSERT_FALSE(depth.empty());
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 272.0; K(1, 1) = 272.0; K(0, 2) = 272.0; K(1, 2) = 240.0;

    // pixels taken from the fixture's first depth plane corners + centre
    Eigen::MatrixX2d keypoints(4, 2);
    keypoints << 0.0, 0.0, 15.0, 11.0, 7.6, 5.2, -3.0, 4.0;  // last is out of bounds
    const auto [points_in_world, inliers] =
        tinynav::mapping::keypoint_with_depth_to_3d(keypoints, depth, pose, K);
    ASSERT_EQ(points_in_world.rows(), 4);
    ASSERT_EQ(inliers.size(), 4u);
    EXPECT_TRUE(inliers[0]);
    EXPECT_TRUE(inliers[1]);
    EXPECT_TRUE(inliers[2]);
    EXPECT_FALSE(inliers[3]);  // u=-3 out of bounds -> invalid

    // hand-computed inlier #3 (7.6, 5.2): int() truncation -> pixel (7,5)
    const double z = depth.at<float>(5, 7);
    ASSERT_GT(z, 0.0);
    const double x = (7 - 272.0) * z / 272.0;
    const double y = (5 - 240.0) * z / 272.0;
    const Eigen::Vector3d expected_world =
        pose.topLeftCorner<3, 3>() * Eigen::Vector3d(x, y, z) +
        pose.topRightCorner<3, 1>();
    EXPECT_NEAR(points_in_world(2, 0), expected_world[0], 1e-9);
    EXPECT_NEAR(points_in_world(2, 1), expected_world[1], 1e-9);
    EXPECT_NEAR(points_in_world(2, 2), expected_world[2], 1e-9);
}

TEST(map_v2_alignment, missing_dir_degrades) {
    tinynav::mapping::MapV2 map;
    std::string error;
    EXPECT_FALSE(tinynav::mapping::load_map_v2("fixtures/map_v2/does_not_exist", map, error));
    EXPECT_FALSE(error.empty());
}

TEST(map_v2_alignment, lazy_views_non_owning_and_missing_degrade) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/map_v2 not exported; run tools/export_map_v2_fixtures.py";
    }
    tinynav::mapping::MapV2 map;
    std::string error;
    ASSERT_TRUE(tinynav::mapping::load_map_v2("fixtures/map_v2/v2", map, error)) << error;

    // Features must be views over the mmap, never per-keyframe copies: the
    // Mat data pointers must land exactly on the mapped file rows. Depth may
    // legitimately return an owning Mat — u16 mm converts to f32 meters on
    // touch (the f4 zero-copy branch is covered by the f32-map e2e).
    tinynav::mapping::MapV2Features feat;
    ASSERT_TRUE(map.get_features(map.timestamps[0], feat));
    ASSERT_FALSE(feat.descps.empty());
    EXPECT_EQ(static_cast<const void*>(feat.descps.data),
              map.feature_descps_.row(
                  static_cast<size_t>(map.feature_offsets_[0])))
        << "descps must be an mmap view";
    const cv::Mat depth = map.get_depth(map.timestamps[0]);
    ASSERT_FALSE(depth.empty());
    EXPECT_EQ(depth.type(), CV_32F);
    EXPECT_TRUE(map.has_frame(map.timestamps[0]));

    // Unknown timestamps degrade exactly like the old .find() == end() path.
    const int64_t absent = -123;
    EXPECT_FALSE(map.has_frame(absent));
    EXPECT_TRUE(map.get_depth(absent).empty());
    tinynav::mapping::MapV2Features absent_feat;
    EXPECT_FALSE(map.get_features(absent, absent_feat));
}

namespace {
long vm_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::strtol(line.c_str() + 6, nullptr, 10);
        }
    }
    return -1;
}
}  // namespace

// The reason the loader went lazy: a yishang-scale map holds >10G of depth
// planes; eager materialization OOMs every Jetson that isn't an AGX. Generate
// the fixture with tools/make_stress_map_v2.py — this test asserts loading
// such a map keeps RSS near-baseline and touching a few candidate frames
// faults only those pages.
TEST(map_v2_alignment, stress_map_stays_lazy) {
    if (!std::filesystem::exists("fixtures/map_v2_stress/v2/pose_timestamps.npy")) {
        GTEST_SKIP() << "fixtures/map_v2_stress not generated; run tools/make_stress_map_v2.py";
    }
    const long before = vm_rss_kb();
    tinynav::mapping::MapV2 map;
    std::string error;
    ASSERT_TRUE(tinynav::mapping::load_map_v2("fixtures/map_v2_stress/v2", map, error)) << error;
    const long loaded = vm_rss_kb();
    const double depth_gb =
        static_cast<double>(map.depth_images_.file_bytes()) / (1024.0 * 1024.0 * 1024.0);
    ASSERT_GT(depth_gb, 2.0) << "stress fixture too small to prove laziness";
    EXPECT_LT(loaded - before, 1'000'000L)  // < 1G while the file holds >2G
        << "map load materialized the arrays — laziness regressed";

    // Touch every 8th frame (reloc's real pattern is a few candidates per
    // query): only those pages may fault in.
    const size_t stride = map.timestamps.size() / 8 + 1;
    for (size_t i = 0; i < map.timestamps.size(); i += stride) {
        const cv::Mat depth = map.get_depth(map.timestamps[i]);
        ASSERT_FALSE(depth.empty());
        double sink = 0.0;
        for (int r = 0; r < depth.rows; r += 37) {
            sink += depth.at<float>(r, r % depth.cols);
        }
        tinynav::mapping::MapV2Features feat;
        ASSERT_TRUE(map.get_features(map.timestamps[i], feat));
        EXPECT_FALSE(feat.kpts.empty());
    }
    const long touched = vm_rss_kb();
    EXPECT_LT(touched - loaded, 500'000L)  // < 500M for 8 touched frames
        << "touching a few frames faulted in the whole file";
}
