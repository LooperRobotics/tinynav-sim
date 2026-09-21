// Alignment test for the perception window factor graph: the scenario in
// fixtures/gtsam_refine is solved by the python gtsam bindings
// (tools/export_gtsam_fixtures.py, the tool-side port of the [ISAM Processing]
// block); this test runs tinynav::gtsam::Refine::refine() on the same numbers
// and expects the same solution (same gtsam build on both sides). Tests SKIP
// when fixtures/ is absent or the build has no GTSAM.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/gtsam/refine.hpp"

namespace {

bool fixtures_available() {
    return std::filesystem::exists("fixtures/gtsam_refine/expected_poses.npy");
}

// minimal npy reader (f64/i64, C-order) — same contract as the other fixtures
bool load_npy(const std::string& path, std::vector<int64_t>& shape,
              std::vector<double>& data, bool* is_i64 = nullptr) {
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
    const bool is_f64 = header.find("<f8", descr_pos) != std::string::npos;
    const bool is_i8 = header.find("<i8", descr_pos) != std::string::npos;
    if (!is_f64 && !is_i8) return false;
    if (is_i64 != nullptr) *is_i64 = is_i8;
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
    if (is_i8) {
        std::vector<int64_t> raw(total);
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total * sizeof(int64_t)));
        for (size_t i = 0; i < total; ++i) data[i] = static_cast<double>(raw[i]);
    } else {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(double)));
    }
    return static_cast<bool>(in);
}

}  // namespace

TEST(gtsam_refine_alignment, window_graph_matches_python) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/gtsam_refine not exported; run tools/export_gtsam_fixtures.py";
    }
    std::shared_ptr<tinynav::gtsam::Refine> refine = tinynav::gtsam::make_gtsam_refine();
    if (refine == nullptr || !refine->available()) {
        GTEST_SKIP() << "built without GTSAM";
    }

    std::vector<int64_t> shape;
    std::vector<double> data;
    const auto load = [&](const char* name) {
        EXPECT_TRUE(load_npy(std::string("fixtures/gtsam_refine/") + name, shape, data));
        return data;
    };

    load("imu_stamps.npy");
    const std::vector<double> imu_stamps = data;
    load("imu_accel.npy");
    const std::vector<double> imu_accel = data;
    load("imu_gyro.npy");
    const std::vector<double> imu_gyro = data;
    load("kf_timestamps.npy");
    const std::vector<double> kf_ts = data;
    load("init_poses.npy");
    const std::vector<double> init_poses = data;
    load("init_velocities.npy");
    const std::vector<double> init_vel = data;
    load("K.npy");
    const std::vector<double> K = data;

    std::vector<tinynav::core::ImuSample> imu;
    for (size_t i = 0; i < imu_stamps.size(); ++i) {
        imu.push_back(tinynav::core::ImuSample{
            imu_stamps[i],
            Eigen::Vector3d(imu_gyro[i * 3], imu_gyro[i * 3 + 1], imu_gyro[i * 3 + 2]),
            Eigen::Vector3d(imu_accel[i * 3], imu_accel[i * 3 + 1], imu_accel[i * 3 + 2])});
    }
    std::vector<Eigen::Matrix4d> poses;
    for (size_t i = 0; i < kf_ts.size(); ++i) {
        Eigen::Matrix4d T;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) T(r, c) = init_poses[i * 16 + r * 4 + c];
        }
        poses.push_back(T);
    }
    std::vector<Eigen::Vector3d> velocities;
    for (size_t i = 0; i < kf_ts.size(); ++i) {
        velocities.emplace_back(init_vel[i * 3], init_vel[i * 3 + 1], init_vel[i * 3 + 2]);
    }

    // ragged tracks -> flat observations + offsets
    std::vector<int64_t> offsets_shape;
    std::vector<double> offsets_data;
    ASSERT_TRUE(load_npy("fixtures/gtsam_refine/track_offsets.npy", offsets_shape, offsets_data,
                         nullptr));
    bool is_i64 = false;
    ASSERT_TRUE(load_npy("fixtures/gtsam_refine/track_observations.npy", shape, data, &is_i64));
    ASSERT_FALSE(is_i64);
    std::vector<tinynav::gtsam::SmartObservation> flat_obs(data.size() / 4);
    for (size_t i = 0; i < flat_obs.size(); ++i) {
        flat_obs[i] = {static_cast<int>(data[i * 4]), data[i * 4 + 1], data[i * 4 + 2],
                       data[i * 4 + 3]};
    }
    tinynav::gtsam::RefineInput input;
    input.keyframe_timestamps.assign(kf_ts.begin(), kf_ts.end());
    input.imu = imu;
    for (size_t t = 0; t + 1 < offsets_data.size(); ++t) {
        const int64_t begin = static_cast<int64_t>(offsets_data[t]);
        const int64_t end = static_cast<int64_t>(offsets_data[t + 1]);
        if (end - begin >= 2) {
            input.tracks.emplace_back(flat_obs.begin() + begin, flat_obs.begin() + end);
        }
    }
    ASSERT_TRUE(load_npy("fixtures/gtsam_refine/velocity_prior_pose_idx.npy", shape, data,
                         &is_i64));
    for (double v : data) input.velocity_prior_pose_idx.push_back(static_cast<int>(v));
    input.K = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K.data());
    load("baseline.npy");
    input.baseline = data[0];

    const tinynav::gtsam::Refine::Metrics metrics = refine->refine(poses, velocities, input);

  printf("metrics: initial=%.9f final=%.9f factors=%d vars=%d tracks=%d\n",
         metrics.initial_error, metrics.final_error, metrics.num_factors,
         metrics.num_variables, metrics.num_tracks);

    // expected solution
    EXPECT_TRUE(load_npy("fixtures/gtsam_refine/expected_poses.npy", shape, data));
    for (size_t i = 0; i < poses.size(); ++i) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                EXPECT_NEAR(poses[i](r, c), data[i * 16 + r * 4 + c], 1e-8) << i << " " << r << c;
            }
        }
    }
    EXPECT_TRUE(load_npy("fixtures/gtsam_refine/expected_velocities.npy", shape, data));
    for (size_t i = 0; i < velocities.size(); ++i) {
        EXPECT_NEAR(velocities[i][0], data[i * 3], 1e-8);
        EXPECT_NEAR(velocities[i][1], data[i * 3 + 1], 1e-8);
        EXPECT_NEAR(velocities[i][2], data[i * 3 + 2], 1e-8);
    }
    EXPECT_TRUE(load_npy("fixtures/gtsam_refine/expected_errors.npy", shape, data));
    EXPECT_NEAR(metrics.initial_error, data[0], 1e-6);
    EXPECT_NEAR(metrics.final_error, data[1], 1e-6);
    EXPECT_TRUE(load_npy("fixtures/gtsam_refine/expected_counts.npy", shape, data, &is_i64));
    EXPECT_EQ(metrics.num_factors, static_cast<int>(data[0]));
    EXPECT_EQ(metrics.num_variables, static_cast<int>(data[1]));
    EXPECT_EQ(metrics.num_tracks, static_cast<int>(data[2]));
    EXPECT_EQ(metrics.num_keyframes, static_cast<int>(kf_ts.size()));
}
