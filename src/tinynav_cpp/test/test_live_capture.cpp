// LiveCapture alignment tests: the nav_temp_db replacement must keep session
// RSS flat (python kept the same data in a disk shelve), return byte-faithful
// features/depth on read, and leave behind a directory that load_map_v2 can
// consume — "the live session files ARE a v2 map subset" is the property the
// online-build plan depends on.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "tinynav_cpp/mapping/live_capture.hpp"
#include "tinynav_cpp/mapping/map_v2.hpp"

namespace {

constexpr int kH = 24, kW = 32, kDesc = 64;

cv::Mat make_depth(unsigned seed) {
    cv::Mat depth(kH, kW, CV_32F);
    for (int r = 0; r < kH; ++r) {
        for (int c = 0; c < kW; ++c) {
            depth.at<float>(r, c) = 0.5f + static_cast<float>(
                ((seed * 131 + r * 31 + c * 7) % 1000)) * 0.01f;  // 0.5..10.5m
        }
    }
    return depth;
}

tinynav::mapping::MapV2Features make_features(unsigned seed, int rows) {
    tinynav::mapping::MapV2Features f;
    const int sz_k[3] = {1, rows, 2};
    f.kpts = cv::Mat(3, sz_k, CV_32F);
    const int sz_d[3] = {1, rows, kDesc};
    f.descps = cv::Mat(3, sz_d, CV_32F);
    const int sz_m[3] = {1, rows, 1};
    f.mask = cv::Mat(3, sz_m, CV_8U);
    for (int j = 0; j < rows; ++j) {
        f.kpts.at<float>(0, j, 0) = static_cast<float>((seed * 17 + j) % kW);
        f.kpts.at<float>(0, j, 1) = static_cast<float>((seed * 23 + j) % kH);
        f.mask.at<uint8_t>(0, j, 0) = static_cast<uint8_t>(j % 2);
        for (int d = 0; d < kDesc; ++d) {
            f.descps.at<float>(0, j, d) = static_cast<float>((seed * 7 + j * 13 + d) % 256) / 255.0f;
        }
    }
    return f;
}

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

TEST(live_capture, append_read_roundtrip_rss_flat) {
    tinynav::mapping::LiveCapture live;
    live.set_dir("fixtures/live_capture_smoke");
    std::string err;

    const long before = vm_rss_kb();
    const int kFrames = 500;
    const int64_t base_ns = 1'726'000'000'000'000'000;
    std::vector<float> first_depth, first_desc;
    int first_with_feats = -1;
    for (int i = 0; i < kFrames; ++i) {
        const int rows = (i % 3 == 0) ? 0 : 8 + (i % 24);  // some empty frames
        auto feats = make_features(static_cast<unsigned>(i), rows);
        cv::Mat depth = make_depth(static_cast<unsigned>(i));
        if (rows > 0 && first_with_feats < 0) {
            first_with_feats = i;
            first_depth.assign(depth.begin<float>(), depth.end<float>());
            first_desc.assign(feats.descps.begin<float>(), feats.descps.end<float>());
        }
        ASSERT_TRUE(live.append(base_ns + i * 66'000'000LL, depth, feats.kpts,
                                feats.descps, feats.mask)) << err;
    }
    ASSERT_EQ(static_cast<int>(live.size()), kFrames);
    ASSERT_GE(first_with_feats, 0);
    const int64_t first_ts = base_ns + first_with_feats * 66'000'000LL;

    // Session RSS must not track the appended payload (this is the property
    // the in-memory maps it replaced lacked).
    const long after = vm_rss_kb();
    EXPECT_LT(after - before, 200'000L) << "live capture grew RAM with the session";

    // Reads come back byte-faithful: depth through the u16-mm round trip
    // (±0.5mm), features as exact f32/u8 values.
    tinynav::mapping::MapV2Features feat;
    ASSERT_TRUE(live.get_features(first_ts, feat));
    ASSERT_EQ(feat.descps.dims, 3);
    ASSERT_EQ(feat.descps.size[1], 8 + (first_with_feats % 24));
    ASSERT_EQ(feat.descps.size[2], kDesc);
    for (size_t i = 0; i < first_desc.size(); ++i) {
        ASSERT_FLOAT_EQ(feat.descps.at<float>(0, static_cast<int>(i / kDesc),
                                              static_cast<int>(i % kDesc)),
                        first_desc[i]);
    }
    const cv::Mat depth = live.get_depth(first_ts);
    ASSERT_FALSE(depth.empty());
    for (int r = 0; r < kH; ++r) {
        for (int c = 0; c < kW; ++c) {
            EXPECT_NEAR(depth.at<float>(r, c), first_depth[static_cast<size_t>(r * kW + c)],
                        5.1e-4);
        }
    }
    // unknown ts degrades
    EXPECT_FALSE(live.has(base_ns - 1));
    EXPECT_TRUE(live.get_depth(base_ns - 1).empty());

    live.close();

    // Close patches the streaming files into a valid v2 subset: feed them to
    // the real reader with synthesized poses/VLAD and it must load.
    std::filesystem::create_directories("fixtures/live_capture_smoke");
    const int n = kFrames;
    std::vector<int64_t> ts(n);
    for (int i = 0; i < n; ++i) {
        ts[i] = base_ns + i * 66'000'000LL;
    }
    auto npy = [](const std::string& path, const char* descr,
                  const std::vector<int64_t>& shape, const void* data, size_t bytes) {
        std::string dims;
        for (size_t i = 0; i < shape.size(); ++i) dims += (i ? "," : "") + std::to_string(shape[i]);
        if (shape.size() == 1) dims += ",";
        std::string h = std::string("{'descr': '") + descr +
                        "', 'fortran_order': False, 'shape': (" + dims + ",), }";
        h.resize(118, ' ');
        h.push_back('\n');
        std::string header(10, '\0');
        header[0] = char(0x93);
        header[1] = 'N'; header[2] = 'U'; header[3] = 'M'; header[4] = 'P'; header[5] = 'Y';
        header[6] = 1; header[7] = 0;
        const uint16_t len = static_cast<uint16_t>(h.size());
        std::memcpy(header.data() + 8, &len, 2);
        std::ofstream out(path, std::ios::binary);
        out.write(header.data(), 10);
        out.write(h.data(), static_cast<std::streamsize>(h.size()));
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };
    npy("fixtures/live_capture_smoke/pose_timestamps.npy", "<i8", {n}, ts.data(),
        ts.size() * sizeof(int64_t));
    std::vector<double> poses(n * 16, 0.0);
    for (int i = 0; i < n; ++i) {
        poses[i * 16 + 0] = 1.0;
        poses[i * 16 + 5] = 1.0;
        poses[i * 16 + 10] = 1.0;
        poses[i * 16 + 15] = 1.0;
        poses[i * 16 + 3] = i * 0.3;
    }
    npy("fixtures/live_capture_smoke/pose_matrices.npy", "<f8", {n, 4, 4}, poses.data(),
        poses.size() * sizeof(double));
    std::vector<float> centres(32, 0.5f);
    npy("fixtures/live_capture_smoke/vlad_centres.npy", "<f4", {4, 8}, centres.data(),
        centres.size() * sizeof(float));
    std::vector<float> descs(n * 8, 0.5f);
    npy("fixtures/live_capture_smoke/vlad_descriptors.npy", "<f4", {n, 8}, descs.data(),
        descs.size() * sizeof(float));

    tinynav::mapping::MapV2 map;
    ASSERT_TRUE(tinynav::mapping::load_map_v2("fixtures/live_capture_smoke", map, err)) << err;
    ASSERT_EQ(map.timestamps.size(), static_cast<size_t>(n));
    tinynav::mapping::MapV2Features map_feat;
    ASSERT_TRUE(map.get_features(ts[static_cast<size_t>(first_with_feats)], map_feat));
    for (size_t i = 0; i < first_desc.size(); ++i) {
        ASSERT_FLOAT_EQ(map_feat.descps.at<float>(0, static_cast<int>(i / kDesc),
                                                  static_cast<int>(i % kDesc)),
                        first_desc[i]);
    }
}

TEST(live_capture, shape_change_fails_frame_not_session) {
    tinynav::mapping::LiveCapture live;
    live.set_dir("fixtures/live_capture_shape");
    std::string err;
    const int64_t ts0 = 1'000'000'000LL;
    cv::Mat depth0 = make_depth(1);
    auto f0 = make_features(1, 10);
    ASSERT_TRUE(live.append(ts0, depth0, f0.kpts, f0.descps, f0.mask));
    // mismatched depth shape on the next frame: the frame is rejected, the
    // session stays usable and the first frame stays readable
    cv::Mat bad(12, 18, CV_32F, 1.0f);
    EXPECT_FALSE(live.append(ts0 + 1, bad, f0.kpts, f0.descps, f0.mask));
    EXPECT_TRUE(live.has(ts0));
    EXPECT_FALSE(live.has(ts0 + 1));
    EXPECT_FALSE(live.get_depth(ts0).empty());
    live.close();
}
