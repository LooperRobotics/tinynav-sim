// Port of reference/tinynav/core/map_node.py::MapNode — the ROS shell and
// state machine (keyframe mapping → loop closure → pose graph, relocalization
// fusion, POI navigation, capture-path priors). Kernels come from the ported
// libraries: mapping/ (VLAD, FusionWindow, PathSpeed/PathClimb, A*) and
// kernels/ (pose_graph_solve), trt/ (SuperPoint, LightGlue, DINOv2).
//
// Deliberate v1 divergences (per README "Known gaps" — map format v2 pending):
//  - poses.npy (a pickled dict) and the TinyNavDB shelve store (VLAD centres +
//    map descriptors + reference features) are unreadable from C++ until the
//    v2 exporter exists. Loading them degrades with a loud warning and
//    disables keyframe relocalization — which in turn keeps
//    T_from_map_to_odom unset, so nav_target_timer early-returns exactly as
//    the Python does when its own guards fail. Everything else (keyframe
//    mapping, loop closure, pose-graph trajectory, priors when their .npy
//    files exist) runs on the ported path.
//  - nav_temp_db (a scratch TinyNavDB) becomes the in-memory stores below —
//    it was a write-through cache for this node's own keyframes.
//  - OdomPoseRecorder (benchmark logging) is not ported.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

// Jazzy's cv_bridge 4.x renamed the header to .hpp (Humble only has .h).
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <opencv2/imgcodecs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>

#include "tinynav_cpp/core/math.hpp"
#include "tinynav_cpp/kernels/pose_graph_solver.hpp"
#include "tinynav_cpp/mapping/astar.hpp"
#include "tinynav_cpp/mapping/fusion_window.hpp"
#include "tinynav_cpp/mapping/live_capture.hpp"
#include "tinynav_cpp/mapping/map_v2.hpp"
#include "tinynav_cpp/mapping/path_prior.hpp"
#include "tinynav_cpp/mapping/vlad.hpp"
#include "tinynav_cpp/trt/models.hpp"

namespace tinynav {

class MappingComponent : public rclcpp::Node {
  public:
    ~MappingComponent() override { live_.close(); }  // flush the live npys
    explicit MappingComponent(const rclcpp::NodeOptions& options)
        : Node("map_node", options) {
        std::string model_dir = "/tinynav/tinynav/models";
        std::string map_path;
        std::string live_capture_dir;
        declare_parameter<std::string>("model_dir", model_dir);
        declare_parameter<std::string>("map_path", map_path);
        declare_parameter<bool>("climb_prior", true);
        // scratch lives under the gitignored fixtures/ — a run from the repo
        // root must never drop an untracked dir into the work tree
        declare_parameter<std::string>("live_capture_dir", "fixtures/nav_temp_v2");
        model_dir = get_parameter("model_dir").as_string();
        map_path = get_parameter("map_path").as_string();
        climb_prior_ = get_parameter("climb_prior").as_bool();
        live_.set_dir(get_parameter("live_capture_dir").as_string());

        super_point_extractor_ = std::make_unique<trt::SuperPointTRT>(model_dir);
        light_glue_matcher_ = std::make_unique<trt::LightGlueTRT>(model_dir);
        dinov2_model_ = std::make_unique<trt::Dinov2TRT>(model_dir);

        // Failure-site dump channel for reloc debugging (see
        // reloc_dump_failure). Set TINYNAV_RELOC_DUMP_DIR in the environment
        // of the stack process (e.g. before run_simulator.sh) to enable.
        if (const char* dump_dir = std::getenv("TINYNAV_RELOC_DUMP_DIR");
            dump_dir != nullptr && dump_dir[0] != '\0') {
            reloc_dump_dir_ = dump_dir;
            RCLCPP_INFO(get_logger(), "reloc failure dumps enabled: %s",
                        reloc_dump_dir_.c_str());
        }

        depth_sub_.subscribe(this, "/slam/keyframe_depth");
        keyframe_image_sub_.subscribe(this, "/slam/keyframe_image");
        keyframe_odom_sub_.subscribe(this, "/slam/keyframe_odom");
        sync_ = std::make_shared<Sync>(10, keyframe_image_sub_, keyframe_odom_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&MappingComponent::keyframe_callback, this,
                                          std::placeholders::_1, std::placeholders::_2,
                                          std::placeholders::_3));
        continuous_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/slam/odometry", 100,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { continuous_odom_callback(*msg); });
        pois_sub_ = create_subscription<std_msgs::msg::String>(
            "/mapping/cmd_pois", 10,
            [this](std_msgs::msg::String::ConstSharedPtr msg) { pois_callback(*msg); });

        pose_graph_trajectory_pub_ = create_publisher<nav_msgs::msg::Path>("/mapping/pose_graph_trajectory", 10);
        relocation_pub_ = create_publisher<nav_msgs::msg::Odometry>("/map/relocalization", 10);
        current_pose_in_map_pub_ = create_publisher<nav_msgs::msg::Odometry>("/mapping/current_pose_in_map", 10);
        speed_cap_pub_ = create_publisher<std_msgs::msg::Float32>("/planning/speed_cap", 10);
        climb_region_pub_ = create_publisher<sensor_msgs::msg::PointCloud>("/planning/climb_region", 10);
        on_stairs_pub_ = create_publisher<std_msgs::msg::Bool>("/planning/on_stairs", 10);
        localization_data_saved_pub_ = create_publisher<std_msgs::msg::Bool>("/benchmark/data_saved", 10);

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/infra2/camera_info", 10,
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { info_callback(*msg); });

        load_map(map_path);

        poi_pub_ = create_publisher<nav_msgs::msg::Odometry>("/mapping/poi", 10);
        poi_change_pub_ = create_publisher<nav_msgs::msg::Odometry>("/mapping/poi_change", 10);
        nav_done_pub_ = create_publisher<std_msgs::msg::Bool>("/mapping/nav_done", 10);
        nav_progress_pub_ = create_publisher<std_msgs::msg::String>("/mapping/nav_progress", 10);
        current_pose_pub_ = create_publisher<nav_msgs::msg::Odometry>("/mapping/current_pose", 10);
        global_plan_pub_ = create_publisher<nav_msgs::msg::Path>("/mapping/global_plan", 10);
        target_pose_pub_ = create_publisher<nav_msgs::msg::Odometry>("/control/target_pose", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        nav_target_timer_ = rclcpp::create_timer(
            this, get_clock(), std::chrono::milliseconds(500),
            [this] { nav_target_timer_callback(); });
        map_prior_timer_ = rclcpp::create_timer(
            this, get_clock(),
            std::chrono::milliseconds(static_cast<int64_t>(1000.0 / kMapPriorHz)),
            [this] { tick_map_priors(); });
    }

  private:
    using Image = sensor_msgs::msg::Image;

    static constexpr double kLoopSimilarityThreshold = 0.90;
    static constexpr int kLoopTopK = 1;
    // select_relocalization_candidates: threshold -1.0 (keep all), top-k best last.
    static constexpr double kRelocSimilarityThreshold = -1.0;
    static constexpr int kRelocLoopTopK = 3;  // relocalization_loop_top_k
    static constexpr int kRelocalizationLoopTopK = 3;
    static constexpr double kArriveM = 0.5;
    static constexpr double kArriveHeadingM = 0.5;
    static constexpr int kArriveTicks = 2;  // TINYNAV_ARRIVE_TICKS default
    static constexpr double kLookaheadS = 5.0;
    static constexpr double kNoCapSpeedMps = 0.6;
    static constexpr double kLookaheadMinM = 1.0;
    static constexpr double kLookaheadMaxM = 5.0;
    static constexpr double kMapPriorHz = 2.0;
    static constexpr double kClimbRegionCullM = 3.5;
    // One keyframe's SuperPoint output (the Python dict).
    struct Features {
        cv::Mat kpts;     // [1, 2, N] CV_32F
        cv::Mat descps;   // [1, N, D]
        cv::Mat mask;     // [1, N, 1]
    };

    // ---------------------------------------------------------------- npy I/O
    // Minimal .npy reader for plain numeric arrays (the pickled poses dict is
    // deliberately NOT handled — see the header comment). Replaces the four
    // np.load calls in load_map until map format v2.
    static bool load_npy(const std::string& path, std::vector<int64_t>& shape,
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
        } else if (ver[0] == 2 || ver[0] == 3) {
            uint32_t len32 = 0;
            in.read(reinterpret_cast<char*>(&len32), 4);
            header_len = static_cast<uint16_t>(len32);
        } else {
            return false;
        }
        std::string header(header_len, '\0');
        in.read(header.data(), header_len);
        // Parse the dict fields the loader needs: descr / fortran_order / shape.
        const auto descr_pos = header.find("'descr'");
        const auto shape_pos = header.find("'shape'");
        if (descr_pos == std::string::npos || shape_pos == std::string::npos) return false;
        const bool is_f32 = header.find("<f4", descr_pos) != std::string::npos;
        const bool is_f64 = header.find("<f8", descr_pos) != std::string::npos;
        if (!is_f32 && !is_f64) return false;
        const bool fortran = header.find("True", header.find("'fortran_order'")) != std::string::npos;
        if (fortran) return false;  // every writer here saves C-order

        shape.clear();
        const auto open = header.find('(', shape_pos);
        const auto close = header.find(')', shape_pos);
        if (open == std::string::npos || close == std::string::npos) return false;
        std::stringstream ss(header.substr(open + 1, close - open - 1));
        std::string token;
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
            if (!in) return false;
            for (size_t i = 0; i < total; ++i) data[i] = raw[i];
        } else {
            in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(double)));
            if (!in) return false;
        }
        return true;
    }

    // ---------------------------------------------------------------- loading
    // Port of load_map + load_map_priors, minus what map format v2 has to carry.
    void load_map(const std::string& map_path) {
        if (map_path.empty()) {
            RCLCPP_WARN(get_logger(),
                        "map_path is empty: no map loaded, relocalization and global planning stay disabled");
            relocalization_enabled_ = false;
            return;
        }
        std::vector<int64_t> shape;
        std::vector<double> data;
        if (load_npy(map_path + "/occupancy_grid.npy", shape, data) && shape.size() == 3) {
            occupancy_map_ = mapping::OccupancyGrid(static_cast<int>(shape[0]),
                                                    static_cast<int>(shape[1]),
                                                    static_cast<int>(shape[2]));
            std::transform(data.begin(), data.end(), occupancy_map_.data().begin(),
                           [](double v) { return static_cast<uint8_t>(v); });
        } else {
            RCLCPP_WARN(get_logger(), "occupancy_grid.npy missing/unreadable: global planning disabled");
        }
        if (load_npy(map_path + "/sdf_map.npy", shape, data) && shape.size() == 3) {
            sdf_map_ = mapping::SdfGrid(static_cast<int>(shape[0]), static_cast<int>(shape[1]),
                                        static_cast<int>(shape[2]));
            std::copy(data.begin(), data.end(), sdf_map_.data().begin());
        } else {
            RCLCPP_WARN(get_logger(), "sdf_map.npy missing/unreadable: global planning disabled");
        }
        if (load_npy(map_path + "/occupancy_meta.npy", shape, data) && shape.size() == 1 &&
            shape[0] >= 4) {
            occupancy_map_origin_ = Eigen::Vector3d(data[0], data[1], data[2]);
            occupancy_resolution_ = data[3];
        } else {
            RCLCPP_WARN(get_logger(), "occupancy_meta.npy missing/unreadable: global planning disabled");
        }
        if (load_npy(map_path + "/intrinsics.npy", shape, data) && shape.size() == 2 &&
            shape[0] == 3 && shape[1] == 3) {
            map_K_ = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(data.data());
        }
        try {
            RCLCPP_INFO(get_logger(), "[speed] %s", bake_message(map_path, "path_speed.npy").c_str());
            speed_index_.emplace(mapping::PathSpeedIndex::load(map_path + "/path_speed.npy"));
        } catch (const std::exception& exc) {
            RCLCPP_ERROR(get_logger(), "[speed] %s/path_speed.npy unusable: %s", map_path.c_str(),
                         exc.what());
            speed_index_.reset();
        }
        if (!climb_prior_) {
            RCLCPP_INFO(get_logger(), "[climb] climb_prior=false — no climb prior, strict everywhere");
            return;
        }
        try {
            RCLCPP_INFO(get_logger(), "[climb] %s", bake_message(map_path, "path_climb.npy").c_str());
            climb_index_.emplace(mapping::PathClimbIndex::load(map_path + "/path_climb.npy"));
            RCLCPP_INFO(get_logger(), "[climb] %d/%zu capture samples labelled climbing",
                        mapping::n_climbing(climb_index_->index().pts()),
                        climb_index_->index().pts().rows());
        } catch (const std::exception& exc) {
            RCLCPP_ERROR(get_logger(), "[climb] %s/path_climb.npy unusable: %s", map_path.c_str(),
                         exc.what());
            climb_index_.reset();
        }
        // Map format v2 (tools/export_map_v2.py) replaces the pickled poses.npy
        // and the VLAD/feature shelve: load it and arm relocalization.
        mapping::MapV2 map_index;
        std::string map_error;
        if (mapping::load_map_v2(map_path, map_index, map_error)) {
            map_index_ = std::move(map_index);
            relocalization_enabled_ = true;
            RCLCPP_INFO(get_logger(),
                        "map v2 loaded: %zu keyframes, VLAD %ldx%ld f32 (%.0fMB), "
                        "depth/features mmap-lazy (%.1fG on disk) — keyframe "
                        "relocalization enabled",
                        map_index_.timestamps.size(),
                        static_cast<long>(map_index_.vlad_descriptors.rows()),
                        static_cast<long>(map_index_.vlad_descriptors.cols()),
                        static_cast<double>(map_index_.vlad_descriptors.size() *
                                            sizeof(float)) /
                          (1024.0 * 1024.0),
                        static_cast<double>(map_index_.depth_images_.file_bytes() +
                                            map_index_.feature_descps_.file_bytes()) /
                          (1024.0 * 1024.0 * 1024.0));
        } else {
            RCLCPP_WARN(get_logger(),
                        "map v2 index not loaded from %s (%s) — keyframe "
                        "relocalization disabled",
                        map_path.c_str(), map_error.c_str());
            relocalization_enabled_ = false;
        }
    }

    // Port of bake_path_speed/bake_path_climb's mtime staleness (log line only;
    // the bake itself is the offline Python toolchain's job).
    static std::string bake_message(const std::string& map_path, const std::string& prior) {
        return prior + ": read (bake is the offline toolchain's job)";
    }

    // ----------------------------------------------------------- callbacks
    void info_callback(const sensor_msgs::msg::CameraInfo& msg) {
        if (K_.has_value()) return;
        Eigen::Matrix3d k;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) k(r, c) = msg.k[static_cast<size_t>(r * 3 + c)];
        }
        K_ = k;
        baseline_ = -msg.p[3] / k(0, 0);
        RCLCPP_INFO(get_logger(), "Camera intrinsics received.");
        camera_info_sub_.reset();
    }

    void continuous_odom_callback(const nav_msgs::msg::Odometry& msg) {
        latest_odom_pose_ = odom_to_T(msg);
    }

    // Port of pois_callback.
    void pois_callback(const std_msgs::msg::String& msg) {
        RCLCPP_INFO(get_logger(), "Received POIs from planner: %s", msg.data.c_str());
        pois_.clear();
        poi_has_heading_.clear();
        // Minimal JSON parse for {"<k>": {"position": [x,y,z], "yaw_deg": a}}.
        if (!parse_pois(msg.data, pois_, poi_has_heading_)) {
            RCLCPP_ERROR(get_logger(), "Failed to parse POIs JSON");
            pois_.clear();
        }
        if (pois_.empty()) {
            poi_index_ = -1;
            cached_nav_path_in_map_.reset();
            nav_msgs::msg::Odometry dummy;
            dummy.header.stamp = now();
            dummy.header.frame_id = "world";
            dummy.child_frame_id = "map";
            dummy.pose.pose.orientation.w = 1.0;
            poi_change_pub_->publish(std::make_unique<nav_msgs::msg::Odometry>(dummy));
            RCLCPP_INFO(get_logger(), "POIs cleared, navigation cancelled");
            return;
        }
        poi_index_ = 0;
        nav_completed_ = false;
        leg_initial_length_.reset();
        leg_start_time_.reset();
        speed_estimate_.reset();
        cached_nav_path_in_map_.reset();
        cached_nav_path_poi_index_ = -1;
    }

    // Port of keyframe_callback.
    void keyframe_callback(Image::ConstSharedPtr keyframe_image_msg,
                           nav_msgs::msg::Odometry::ConstSharedPtr keyframe_odom_msg,
                           Image::ConstSharedPtr depth_msg) {
        keyframe_mapping(*keyframe_image_msg, *keyframe_odom_msg, *depth_msg);
        // keyframe_relocalization needs the map VLAD index (map format v2); the
        // compute_transform_from_map_to_odom it feeds stays dormant with it.
        if (relocalization_enabled_) {
            const cv::Mat image = cv_bridge::toCvCopy(*keyframe_image_msg, "mono8")->image;
            if (keyframe_relocalization(keyframe_image_msg->header, image)) {
                compute_transform_from_map_to_odom();
            }
        }
    }

    // Port of keyframe_mapping (nav_temp_db → in-memory stores).
    void keyframe_mapping(const Image& keyframe_image_msg,
                          const nav_msgs::msg::Odometry& keyframe_odom_msg,
                          const Image& depth_msg) {
        if (!K_.has_value()) return;
        const int64_t keyframe_image_timestamp = stamp_to_ns(keyframe_image_msg.header.stamp);
        const int64_t keyframe_odom_timestamp = stamp_to_ns(keyframe_odom_msg.header.stamp);
        const int64_t depth_timestamp = stamp_to_ns(depth_msg.header.stamp);
        if (keyframe_image_timestamp != keyframe_odom_timestamp ||
            keyframe_image_timestamp != depth_timestamp) {
            RCLCPP_ERROR(get_logger(), "keyframe sync mismatch: %ld / %ld / %ld",
                         static_cast<long>(keyframe_image_timestamp),
                         static_cast<long>(keyframe_odom_timestamp),
                         static_cast<long>(depth_timestamp));
            return;
        }
        const cv::Mat image = cv_bridge::toCvCopy(keyframe_image_msg, "mono8")->image;
        const cv::Mat depth_img = cv_bridge::toCvCopy(depth_msg, "32FC1")->image;
        const Eigen::Matrix4d odom = odom_to_T(keyframe_odom_msg);

        Features features;
        if (super_point_extractor_->infer(image, superpoint_results_)) {
            features.kpts = superpoint_results_["kpts"];
            features.descps = superpoint_results_["descps"];
            const auto it = superpoint_results_.find("mask");
            if (it != superpoint_results_.end()) features.mask = it->second;
            latest_keyframe_features_ = {keyframe_image_timestamp, features};
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "superpoint engine unavailable: mapping degraded");
        }

        // nav_temp_db parity: depth + features land in the on-disk scratch
        // store (u16-mm depth), loop closure reads them back per candidate —
        // per-session RSS no longer scales with keyframe count. A frame whose
        // append failed is simply invisible to loop closure.
        if (!live_.append(keyframe_image_timestamp, depth_img, features.kpts,
                          features.descps, features.mask)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "live capture append failed: keyframe %ld stays "
                                 "out of loop closure",
                                 static_cast<long>(keyframe_image_timestamp));
        } else if (dinov2_model_->available()) {
            // Embedding only when the frame actually persisted: find_loop's
            // candidates must never point at a missing store entry.
            std::vector<float> embedding;
            if (dinov2_model_->infer(image, embedding)) {
                embeddings_[keyframe_image_timestamp] =
                    Eigen::Map<Eigen::VectorXf>(embedding.data(),
                                                static_cast<Eigen::Index>(embedding.size()))
                        .cast<double>();
            }
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "dinov2 engine unavailable: loop closure disabled");
        }

        if (odom_.empty() && !last_keyframe_timestamp_.has_value()) {
            odom_[keyframe_odom_timestamp] = odom;
            pose_graph_used_pose_[keyframe_image_timestamp] = odom;
        } else {
            if (!last_keyframe_timestamp_.has_value()) return;
            const Eigen::Matrix4d last = odom_.at(*last_keyframe_timestamp_);
            const Eigen::Matrix4d T_prev_curr = last.inverse() * odom;
            relative_pose_constraint_.emplace_back(keyframe_image_timestamp,
                                                   *last_keyframe_timestamp_, T_prev_curr);
            pose_graph_used_pose_[keyframe_image_timestamp] = odom;
            odom_[keyframe_image_timestamp] = odom;
            find_loop_and_pose_graph(keyframe_image_timestamp);
            pose_graph_trajectory_publish(keyframe_image_timestamp);
        }
        last_keyframe_timestamp_ = keyframe_odom_timestamp;
        last_keyframe_image_ = image;
    }

    // Port of the inner find_loop_and_pose_graph + build_map_node's find_loop /
    // solve_pose_graph (they were build_map_node imports, not mapping lib calls).
    void find_loop_and_pose_graph(int64_t timestamp) {
        if (embeddings_.find(timestamp) == embeddings_.end()) return;
        const Eigen::VectorXd& target = embeddings_.at(timestamp);
        std::vector<int64_t> valid_timestamp;
        for (const auto& [t, pose] : pose_graph_used_pose_) {
            if (t + 10LL * 1'000'000'000LL < timestamp) valid_timestamp.push_back(t);
        }
        if (valid_timestamp.empty()) return;
        // find_loop: plain dot products, argsort ascending, threshold, last top-k.
        std::vector<std::pair<int, double>> scored;
        for (size_t i = 0; i < valid_timestamp.size(); ++i) {
            const Eigen::VectorXd& emb = embeddings_.at(valid_timestamp[i]);
            scored.emplace_back(static_cast<int>(i), target.dot(emb));
        }
        std::stable_sort(scored.begin(), scored.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        std::vector<std::pair<int, double>> loop_list;
        for (const auto& [idx, sim] : scored) {
            if (sim > kLoopSimilarityThreshold) loop_list.emplace_back(idx, sim);
        }
        const size_t top = static_cast<size_t>(std::max(kLoopTopK, 0));
        if (loop_list.size() > top) {
            loop_list.erase(loop_list.begin(), loop_list.end() - top);
        }
        // Relative pose estimation per loop candidate.
        for (const auto& [idx, similarity] : loop_list) {
            const int64_t prev_timestamp = valid_timestamp[static_cast<size_t>(idx)];
            const int64_t curr_timestamp = timestamp;
            mapping::MapV2Features prev_feat, curr_feat;
            if (!live_.get_features(prev_timestamp, prev_feat) ||
                !live_.get_features(curr_timestamp, curr_feat)) {
                continue;
            }
            auto [prev_matched, curr_matched, matches] =
                match_keypoints(prev_feat, curr_feat);
            if (matches.rows() == 0) continue;
            // u16-mm rows convert back to f32 meters per candidate — event-
            // level cost; the old per-keyframe MatrixXd copy is gone.
            const cv::Mat curr_depth = live_.get_depth(curr_timestamp);
            Eigen::MatrixXd curr_depth_mat;
            if (!curr_depth.empty()) {
                curr_depth_mat.resize(curr_depth.rows, curr_depth.cols);
                curr_depth_mat =
                    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>>(
                        reinterpret_cast<float*>(curr_depth.data), curr_depth.rows,
                        curr_depth.cols)
                        .cast<double>();
            }
            const core::EstimatePoseResult est = core::estimate_pose(
                prev_matched, curr_matched, curr_depth_mat, *K_, {});
            if (est.success && est.inlier_idx_original.size() >= 100) {
                relative_pose_constraint_.emplace_back(curr_timestamp, prev_timestamp,
                                                       est.pose);
            }
        }
        // solve_pose_graph: min timestamp constant, weights 10/10/10 & 30/30/30.
        if (!relative_pose_constraint_.empty()) {
            kernels::CameraPoses cameras;
            for (const auto& [t, pose] : pose_graph_used_pose_) cameras[t] = pose;
            const int64_t min_timestamp =
                std::min_element(cameras.begin(), cameras.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; })
                    ->first;
            kernels::ConstantPoseIndex constant{{min_timestamp, true}};
            std::vector<kernels::RelativePoseConstraint> constraints;
            for (const auto& [curr, prev, T] : relative_pose_constraint_) {
                // build_map_node's tuple is (curr, prev, T_prev_curr) — the same
                // (i, j, relative_pose_j_i) shape the kernel port carries.
                constraints.push_back({curr, prev, T, Eigen::Vector3d(10.0, 10.0, 10.0),
                                       Eigen::Vector3d(30.0, 30.0, 30.0)});
            }
            const kernels::CameraPoses solved =
                kernels::pose_graph_solve(cameras, constraints, constant, 5);
            pose_graph_used_pose_.clear();
            for (const auto& [t, pose] : solved) pose_graph_used_pose_[t] = pose;
        }
    }

    // The Python hardcodes np.array([848,480]) (the real D435 size) at both
    // match_keypoints call sites regardless of the actual keyframe size, and
    // the LightGlue engine's keypoint normalization is sensitive to it —
    // replaying the same pair at 848x480 vs 544x480 moved borderline match
    // counts across the 50-match gate (48→66). Parity requires the same
    // hardcode; class-level so the dump-channel meta records the same value.
    static constexpr std::array<int64_t, 2> kLgImageShape{848, 480};

    // Port of match_keypoints. Templatized so the map-v2 feature struct and the
    // live-keyframe Features (same kpts/descps/mask shape) both fit.
    template <typename F0, typename F1>
    std::tuple<Eigen::MatrixX2d, Eigen::MatrixX2d, Eigen::MatrixX2i> match_keypoints(
        const F0& feats0, const F1& feats1) {
        Eigen::MatrixX2d keypoints0, keypoints1;
        Eigen::MatrixX2i matches(0, 2);
        if (!light_glue_matcher_->available()) return {keypoints0, keypoints1, matches};
        trt::TrtOutputMap out;
        if (!light_glue_matcher_->infer(feats0.kpts, feats1.kpts, feats0.descps,
                                        feats1.descps, feats0.mask, feats1.mask,
                                        kLgImageShape, kLgImageShape, out)) {
            return {keypoints0, keypoints1, matches};
        }
        const cv::Mat& match_indices = out.at("match_indices");  // [1, N] INT32 -> CV_32S
        const int n = match_indices.size[1];
        const Eigen::MatrixX2d k0 = kpts_from_features(feats0);
        const Eigen::MatrixX2d k1 = kpts_from_features(feats1);
        auto match_at = [&](int i) {
            return match_indices.type() == CV_32S ? static_cast<long long>(match_indices.at<int32_t>(0, i))
                                                  : static_cast<long long>(match_indices.at<double>(0, i));
        };
        std::vector<Eigen::Vector2d> p0, p1;
        std::vector<Eigen::Vector2i> m;
        for (int i = 0; i < n; ++i) {
            const long long index = match_at(i);
            if (index != -1) {
                p0.push_back(k0.row(i));
                p1.push_back(k1.row(static_cast<int>(index)));
                m.emplace_back(i, static_cast<int>(index));
            }
        }
        keypoints0.resize(static_cast<int>(p0.size()), 2);
        keypoints1.resize(static_cast<int>(p1.size()), 2);
        for (int i = 0; i < keypoints0.rows(); ++i) {
            keypoints0.row(i) = p0[static_cast<size_t>(i)];
            keypoints1.row(i) = p1[static_cast<size_t>(i)];
        }
        matches.resize(static_cast<int>(m.size()), 2);
        for (int i = 0; i < matches.rows(); ++i) matches.row(i) = m[static_cast<size_t>(i)];
        return {keypoints0, keypoints1, matches};
    }

    template <typename F>
    static Eigen::MatrixX2d kpts_from_features(const F& feats) {
        Eigen::MatrixX2d out;
        if (feats.kpts.empty()) return out;
        const int n = feats.kpts.dims == 3 && feats.kpts.size[1] == 2
                          ? feats.kpts.size[2]
                          : feats.kpts.size[1];
        out.resize(n, 2);
        for (int i = 0; i < n; ++i) {
            if (feats.kpts.dims == 3 && feats.kpts.size[1] == 2) {
                out(i, 0) = feats.kpts.template at<float>(0, 0, i);
                out(i, 1) = feats.kpts.template at<float>(0, 1, i);
            } else {
                out(i, 0) = feats.kpts.template at<float>(0, i, 0);
                out(i, 1) = feats.kpts.template at<float>(0, i, 1);
            }
        }
        return out;
    }

    // Port of pose_graph_trajectory_publish.
    void pose_graph_trajectory_publish(int64_t timestamp) {
        nav_msgs::msg::Path path;
        path.header.stamp.sec = static_cast<int32_t>(timestamp / 1'000'000'000LL);
        path.header.stamp.nanosec = static_cast<uint32_t>(timestamp % 1'000'000'000LL);
        path.header.frame_id = "world";
        for (const auto& [t, pose] : pose_graph_used_pose_) {
            geometry_msgs::msg::PoseStamped stamped;
            stamped.header = path.header;
            const Eigen::Vector3d trans = pose.topRightCorner<3, 1>();
            const Eigen::Vector4d quat = core::matrix_to_quat(pose.topLeftCorner<3, 3>());
            stamped.pose.position.x = trans[0];
            stamped.pose.position.y = trans[1];
            stamped.pose.position.z = trans[2];
            stamped.pose.orientation.x = quat[0];
            stamped.pose.orientation.y = quat[1];
            stamped.pose.orientation.z = quat[2];
            stamped.pose.orientation.w = quat[3];
            path.poses.push_back(stamped);
        }
        pose_graph_trajectory_pub_->publish(std::make_unique<nav_msgs::msg::Path>(std::move(path)));
    }

    // Port of keyframe_relocalization.
    bool keyframe_relocalization(const std_msgs::msg::Header& header, const cv::Mat& image) {
        if (!K_.has_value() || !map_K_.has_value() || map_index_.timestamps.empty()) return false;
        const int64_t timestamp_ns = stamp_to_ns(header.stamp);
        Features features;
        if (latest_keyframe_features_.first == timestamp_ns) {
            // extracted by keyframe_mapping already — no second SuperPoint pass
            features = latest_keyframe_features_.second;
        } else {
            trt::TrtOutputMap results;
            if (!super_point_extractor_->infer(image, results)) return false;
            features.kpts = results["kpts"];
            features.descps = results["descps"];
            const auto mask_it = results.find("mask");
            if (mask_it != results.end()) features.mask = mask_it->second;
        }
        const auto [success, pose_in_camera, pose_cov_weight] =
            relocalize_with_depth(timestamp_ns, image, features);
        if (!success) {
            failed_relocalizations_.push_back(timestamp_ns);
            return false;
        }
        const Eigen::Matrix4d pose_in_world = pose_in_camera.inverse();
        relocation_pub_->publish(
            std::make_unique<nav_msgs::msg::Odometry>(reloc_msg(pose_in_world, header)));
        relocalization_poses_[timestamp_ns] = pose_in_world;
        relocalization_pose_weights_[timestamp_ns] = pose_cov_weight;
        return true;
    }

    // Port of relocalize_with_depth (+ select_relocalization_candidates and
    // rank_relocalization_candidates folded in). timestamp_ns is not in the
    // Python signature — it names the dump-channel failure dirs.
    std::tuple<bool, Eigen::Matrix4d, double> relocalize_with_depth(
        int64_t timestamp_ns, const cv::Mat& image, const Features& features) {
        if (!K_.has_value() || !map_K_.has_value()) {
            return {false, Eigen::Matrix4d::Identity(), -std::numeric_limits<double>::infinity()};
        }
        // query VLAD: compute_vlad(dinov2 patch tokens, map centres)
        cv::Mat tokens;
        if (!dinov2_model_->available() || !dinov2_model_->infer_patch_tokens(image, tokens)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "dinov2 patch tokens unavailable: relocalization skipped");
            return {false, Eigen::Matrix4d::Identity(), -std::numeric_limits<double>::infinity()};
        }
        Eigen::MatrixXd patch_tokens(tokens.rows, tokens.cols);
        for (int r = 0; r < tokens.rows; ++r) {
            for (int c = 0; c < tokens.cols; ++c) {
                patch_tokens(r, c) = tokens.at<float>(r, c);
            }
        }
        const Eigen::VectorXd query_vlad =
            mapping::compute_vlad(patch_tokens, map_index_.vlad_centres);
        // select_relocalization_candidates: find_loop over the map descriptors
        // with threshold -1.0 — top-k by similarity, best LAST. The index is
        // f32 (Orin Nano 8G; python keeps it f32 too) — cast the query once.
        const Eigen::VectorXf query_vlad_f = query_vlad.cast<float>();
        std::vector<std::pair<int, double>> scored;
        for (int i = 0; i < map_index_.vlad_descriptors.rows(); ++i) {
            scored.emplace_back(i, map_index_.vlad_descriptors.row(i).dot(query_vlad_f));
        }
        std::stable_sort(scored.begin(), scored.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        std::vector<std::pair<int, double>> candidates;
        for (const auto& [idx, similarity] : scored) {
            if (similarity > kRelocSimilarityThreshold) candidates.emplace_back(idx, similarity);
        }
        const size_t top = static_cast<size_t>(std::max(kRelocLoopTopK, 0));
        if (candidates.size() > top) {
            candidates.erase(candidates.begin(), candidates.end() - top);
        }
        if (candidates.empty()) {
            RCLCPP_DEBUG(get_logger(), "VLAD: no relocalization candidates");
            return {false, Eigen::Matrix4d::Identity(), -std::numeric_limits<double>::infinity()};
        }

        std::vector<std::pair<Eigen::MatrixX3d, Eigen::MatrixX2d>> pnp_candidates;
        for (const auto& [row, similarity] : candidates) {
            const int64_t ts = map_index_.timestamps[static_cast<size_t>(row)];
            const auto pose_it = map_index_.poses.find(ts);
            if (pose_it == map_index_.poses.end()) {
                continue;
            }
            // Python parity: depth/features come from the map shelve only for
            // the candidate keyframes that reach matching — mmap views here.
            mapping::MapV2Features map_features;
            if (!map_index_.get_features(ts, map_features)) {
                continue;
            }
            const cv::Mat map_depth = map_index_.get_depth(ts);
            if (map_features.kpts.empty() || map_depth.empty()) {
                continue;
            }
            auto [reference_matched_keypoints, keyframe_matched_keypoints, matches] =
                match_keypoints(map_features, features);
            if (matches.rows() < 50) {
                // The Python prints this unconditionally for every rejected
                // candidate; keep it visible (INFO) for field diagnosis.
                RCLCPP_INFO(get_logger(),
                            "not enough matched features to relocalize, %ld < 50 (cand %ld)",
                            static_cast<long>(matches.rows()), static_cast<long>(ts));
                reloc_dump_failure(timestamp_ns, ts, similarity,
                                   static_cast<long>(matches.rows()), image,
                                   map_features.kpts, map_features.descps,
                                   map_features.mask, features.kpts, features.descps,
                                   features.mask);
                continue;
            }
            const auto [point_3d_in_world, inliers] = mapping::keypoint_with_depth_to_3d(
                reference_matched_keypoints, map_depth, pose_it->second, *map_K_);
            std::vector<Eigen::Vector3d> pts3d;
            std::vector<Eigen::Vector2d> pts2d;
            for (int i = 0; i < point_3d_in_world.rows(); ++i) {
                if (inliers[static_cast<size_t>(i)]) {
                    pts3d.push_back(point_3d_in_world.row(i));
                    pts2d.push_back(keyframe_matched_keypoints.row(i));
                }
            }
            if (static_cast<int>(pts2d.size()) <= 80) {
                RCLCPP_DEBUG(get_logger(), "not enough landmarks to relocalize, %zu",
                             pts2d.size());
                continue;
            }
            Eigen::MatrixX3d pts3d_mat(static_cast<int>(pts3d.size()), 3);
            Eigen::MatrixX2d pts2d_mat(static_cast<int>(pts2d.size()), 2);
            for (int i = 0; i < pts3d_mat.rows(); ++i) {
                pts3d_mat.row(i) = pts3d[static_cast<size_t>(i)];
                pts2d_mat.row(i) = pts2d[static_cast<size_t>(i)];
            }
            pnp_candidates.emplace_back(std::move(pts3d_mat), std::move(pts2d_mat));
        }
        if (pnp_candidates.empty()) {
            RCLCPP_DEBUG(get_logger(), "no valid PnP relocalization candidate found");
            return {false, Eigen::Matrix4d::Identity(), -std::numeric_limits<double>::infinity()};
        }
        const core::PnpRerankResult rerank = core::rerank_by_pnp_inliers(pnp_candidates, *K_);
        if (!rerank.success) {
            return {false, Eigen::Matrix4d::Identity(), -std::numeric_limits<double>::infinity()};
        }
        return {true, rerank.pose, rerank.inlier_ratio};
    }

    // Failure-site dump for reloc debugging. Enabled by TINYNAV_RELOC_DUMP_DIR
    // (read in the constructor): every rejected candidate (<50 LightGlue
    // matches) writes the exact LG inputs to <dir>/<live_ts>_<cand_ts>/ as raw
    // f32 files that tools/probes/probe_lg replays directly — the 2026-09-19
    // img_shape root cause was found by replaying such dumps, and rebuilding
    // the throwaway scaffolding each time cost more than the bug itself.
    // Masks are converted to f32 (lossless from u8: the engine sees identical
    // values after copy_to_input). The live keyframe image comes along for
    // eyeballing. Capped at 50 pair dirs so a reloc failure storm (off-map
    // driving) cannot fill the disk. Zero cost when the env var is unset.
    static constexpr int kRelocDumpCap = 50;

    static bool dump_mat_f32(const std::string& path, const cv::Mat& m) {
        cv::Mat f;
        if (m.type() == CV_32F) {
            f = m;
        } else {
            m.convertTo(f, CV_32F);
        }
        if (!f.isContinuous()) f = f.clone();
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(f.data),
                  static_cast<std::streamsize>(f.total()) * sizeof(float));
        return static_cast<bool>(out);
    }

    static const char* mat_depth_name(const cv::Mat& m) {
        switch (m.depth()) {
            case CV_8U: return "u8";
            case CV_32F: return "f32";
            case CV_64F: return "f64";
            default: return "other";
        }
    }

    void reloc_dump_failure(int64_t live_ts, int64_t cand_ts, double similarity,
                            long match_count, const cv::Mat& live_image,
                            const cv::Mat& map_kpts, const cv::Mat& map_descps,
                            const cv::Mat& map_mask, const cv::Mat& live_kpts,
                            const cv::Mat& live_descps, const cv::Mat& live_mask) {
        if (reloc_dump_dir_.empty()) return;
        if (reloc_dump_count_ >= kRelocDumpCap) {
            if (!reloc_dump_cap_logged_) {
                reloc_dump_cap_logged_ = true;
                RCLCPP_WARN(get_logger(),
                            "reloc dump cap %d reached; further failures are not dumped",
                            kRelocDumpCap);
            }
            return;
        }
        const std::string subdir =
            reloc_dump_dir_ + "/" + std::to_string(live_ts) + "_" + std::to_string(cand_ts);
        std::error_code ec;
        std::filesystem::create_directories(subdir, ec);
        if (ec) {
            RCLCPP_WARN(get_logger(), "reloc dump: cannot create %s (%s) — disabling",
                        subdir.c_str(), ec.message().c_str());
            reloc_dump_dir_.clear();
            return;
        }
        struct NamedMat {
            const char* name;
            const cv::Mat& mat;
        };
        const NamedMat mats[] = {{"map_kpts", map_kpts},
                                 {"map_descps", map_descps},
                                 {"map_mask", map_mask},
                                 {"live_kpts", live_kpts},
                                 {"live_descps", live_descps},
                                 {"live_mask", live_mask}};
        bool ok = true;
        for (const auto& [name, mat] : mats) {
            ok = dump_mat_f32(subdir + "/" + name + ".f32", mat) && ok;
        }
        if (ok) {
            std::ofstream meta(subdir + "/meta.json");
            meta << "{\n"
                 << "  \"live_ts\": " << live_ts << ",\n"
                 << "  \"cand_ts\": " << cand_ts << ",\n"
                 << "  \"vlad_similarity\": " << similarity << ",\n"
                 << "  \"match_count\": " << match_count << ",\n"
                 << "  \"gate\": 50,\n"
                 << "  \"lg_img_shape\": [" << kLgImageShape[0] << ", " << kLgImageShape[1]
                 << "],\n"
                 << "  \"map_points\": " << map_kpts.total() / 2 << ",\n"
                 << "  \"live_points\": " << live_kpts.total() / 2 << ",\n"
                 << "  \"map_kpts_type\": \"" << mat_depth_name(map_kpts) << "\",\n"
                 << "  \"live_kpts_type\": \"" << mat_depth_name(live_kpts) << "\",\n"
                 << "  \"map_mask_type\": \"" << mat_depth_name(map_mask) << "\",\n"
                 << "  \"live_mask_type\": \"" << mat_depth_name(live_mask) << "\"\n"
                 << "}\n";
            ok = static_cast<bool>(meta);
        }
        if (ok && !live_image.empty()) {
            cv::imwrite(subdir + "/live.png", live_image);
        }
        ++reloc_dump_count_;
        if (ok) {
            RCLCPP_INFO(get_logger(), "reloc failure dumped to %s", subdir.c_str());
        } else {
            RCLCPP_WARN(get_logger(), "reloc dump to %s failed", subdir.c_str());
        }
    }

    // np2msg(pose, timestamp, "world", "camera").
    static nav_msgs::msg::Odometry reloc_msg(const Eigen::Matrix4d& pose,
                                             const std_msgs::msg::Header& header) {
        nav_msgs::msg::Odometry msg;
        msg.header = header;
        msg.header.frame_id = "world";
        msg.child_frame_id = "camera";
        const Eigen::Vector3d trans = pose.topRightCorner<3, 1>();
        const Eigen::Vector4d quat = core::matrix_to_quat(pose.topLeftCorner<3, 3>());
        msg.pose.pose.position.x = trans[0];
        msg.pose.pose.position.y = trans[1];
        msg.pose.pose.position.z = trans[2];
        msg.pose.pose.orientation.x = quat[0];
        msg.pose.pose.orientation.y = quat[1];
        msg.pose.pose.orientation.z = quat[2];
        msg.pose.pose.orientation.w = quat[3];
        return msg;
    }

    // Port of compute_transform_from_map_to_odom (dormant until v2; kept for
    // the structure and the fusion-window contract it exercises once the
    // relocalization poses exist).
    void compute_transform_from_map_to_odom() {
        if (relocalization_poses_.empty()) return;
        std::vector<kernels::RelativePoseConstraint> constraints;
        std::vector<Eigen::Matrix4d> constraint_odom;
        for (const auto& [timestamp, pose] : relocalization_poses_) {
            const auto used = pose_graph_used_pose_.find(timestamp);
            if (used == pose_graph_used_pose_.end()) continue;
            const Eigen::Matrix4d observation = used->second * pose.inverse();
            const double weight = relocalization_pose_weights_.at(timestamp);
            constraints.push_back({0, 1, observation, Eigen::Vector3d::Constant(10.0 * weight),
                                   Eigen::Vector3d::Constant(10.0 * weight)});
            constraint_odom.push_back(used->second);
        }
        const mapping::FusionWindow window;
        constraints = window.select_fusion_constraints(constraints, constraint_odom);
        if (constraints.empty()) {
            // Nothing observed: the pose must not move (see the Python comment).
            return;
        }
        kernels::CameraPoses cameras{{0, T_from_map_to_odom_.value_or(Eigen::Matrix4d::Identity())},
                                     {1, Eigen::Matrix4d::Identity()}};
        kernels::ConstantPoseIndex constant{{1, true}};
        const kernels::CameraPoses solved =
            kernels::pose_graph_solve(cameras, constraints, constant, 1000);
        T_from_map_to_odom_ = solved.at(0);
    }

    // ------------------------------------------------------- prior tick (2 Hz)
    // Port of tick_map_priors.
    void tick_map_priors() {
        sensor_msgs::msg::PointCloud cloud;
        cloud.header.stamp = now();
        cloud.header.frame_id = "world";
        bool on_stairs = false;
        if (climb_index_.has_value() && T_from_map_to_odom_.has_value() &&
            latest_odom_pose_.has_value()) {
            const Eigen::Matrix4d here_in_map = T_from_map_to_odom_->inverse() * *latest_odom_pose_;
            const Eigen::Vector3d here = here_in_map.topRightCorner<3, 1>();
            on_stairs = climb_index_->on_stairs(here);
            const Eigen::MatrixXd pts_in_map =
                climb_index_->climbing_within(here, kClimbRegionCullM);
            // Guard the empty region: the row-broadcast sum below is UB on a
            // 0-row operand (numpy broadcasts (0,3)+t fine, Eigen segfaults).
            if (pts_in_map.rows() > 0) {
                const Eigen::MatrixXd pts_in_odom =
                    (T_from_map_to_odom_->topLeftCorner<3, 3>() * pts_in_map.transpose())
                            .transpose() +
                    T_from_map_to_odom_->topRightCorner<3, 1>().transpose();
                for (int i = 0; i < pts_in_odom.rows(); ++i) {
                    geometry_msgs::msg::Point32 p;
                    p.x = static_cast<float>(pts_in_odom(i, 0));
                    p.y = static_cast<float>(pts_in_odom(i, 1));
                    p.z = static_cast<float>(pts_in_odom(i, 2));
                    cloud.points.push_back(p);
                }
            }
        }
        climb_region_pub_->publish(std::make_unique<sensor_msgs::msg::PointCloud>(std::move(cloud)));
        auto stairs = std::make_unique<std_msgs::msg::Bool>();
        stairs->data = on_stairs;
        on_stairs_pub_->publish(std::move(stairs));
    }

    // ------------------------------------------------------------ nav timer
    // Port of nav_target_timer_callback (2 Hz).
    void nav_target_timer_callback() {
        if (poi_index_ < 0 || poi_index_ >= static_cast<int>(pois_.size()) ||
            !T_from_map_to_odom_.has_value() || !latest_odom_pose_.has_value()) {
            return;
        }

        const Eigen::Matrix4d pose_in_map = T_from_map_to_odom_->inverse() * *latest_odom_pose_;
        current_pose_in_map_pub_->publish(std::make_unique<nav_msgs::msg::Odometry>(
            to_odom_msg(pose_in_map, now(), "world", "map")));
        const double cap =
            speed_index_.has_value()
                ? speed_index_->speed_cap(pose_in_map.topRightCorner<3, 1>())
                : std::numeric_limits<double>::infinity();
        auto cap_msg = std::make_unique<std_msgs::msg::Float32>();
        cap_msg->data = static_cast<float>(cap);
        speed_cap_pub_->publish(std::move(cap_msg));

        const Eigen::Vector3d poi = pois_[static_cast<size_t>(poi_index_)];
        const Eigen::Vector3d pos = pose_in_map.topRightCorner<3, 1>();
        if (poi_reached(poi, pos)) {
            publish_nav_progress(100.0, 0.0, leg_initial_length_.value_or(0.0), 0.0, true);
            ++poi_index_;
            arrive_ticks_ = 0;
            leg_initial_length_.reset();
            leg_start_time_.reset();
            speed_estimate_.reset();
            cached_nav_path_in_map_.reset();
            cached_nav_path_poi_index_ = -1;
            poi_change_pub_->publish(std::make_unique<nav_msgs::msg::Odometry>(
                to_odom_msg(Eigen::Matrix4d::Identity(), now(), "world", "map")));
            if (poi_index_ >= static_cast<int>(pois_.size()) && !nav_completed_) {
                nav_completed_ = true;
                RCLCPP_INFO(get_logger(), "All POIs have been visited, nav done");
                auto done = std::make_unique<std_msgs::msg::Bool>();
                done->data = true;
                nav_done_pub_->publish(std::move(done));
            }
            return;
        }

        bool needs_replan =
            !cached_nav_path_in_map_.has_value() || cached_nav_path_poi_index_ != poi_index_;
        if (!needs_replan) {
            const Eigen::MatrixXd& paths = *cached_nav_path_in_map_;
            int closest_idx = closest_index(paths, pos);
            if ((paths.row(closest_idx).head<2>().transpose() - pos.head<2>()).norm() > 0.5) {
                needs_replan = true;
            }
        }

        if (needs_replan) {
            const std::optional<Eigen::MatrixXd> paths =
                generate_nav_path_in_map(pose_in_map, poi);
            if (paths.has_value()) {
                cached_nav_path_in_map_ = *paths;
                cached_nav_path_poi_index_ = poi_index_;
            } else {
                cached_nav_path_in_map_.reset();
                cached_nav_path_poi_index_ = -1;
                return;
            }
        }

        const Eigen::MatrixXd& paths = *cached_nav_path_in_map_;
        publish_global_plan(paths);
        const int closest_idx = closest_index(paths, pos);

        double remaining_length = 0.0;
        if (closest_idx < paths.rows() - 1) {
            for (int i = closest_idx; i < paths.rows() - 1; ++i) {
                remaining_length += (paths.row(i + 1) - paths.row(i)).norm();
            }
        }
        const auto [percent, estimated_remaining_s] = update_leg_progress(remaining_length);
        publish_nav_progress(std::round(percent * 10.0) / 10.0,
                             std::round(remaining_length * 100.0) / 100.0,
                             std::round(leg_initial_length_.value_or(0.0) * 100.0) / 100.0,
                             std::round(estimated_remaining_s * 10.0) / 10.0, false);

        const Eigen::Vector3d target_position =
            select_target_position(paths, closest_idx, pos, cap);

        const Eigen::Matrix4d T = *latest_odom_pose_ * pose_in_map.inverse();
        const Eigen::Vector3d target_position_in_odom =
            T.topLeftCorner<3, 3>() * target_position + T.topRightCorner<3, 1>();
        Eigen::Matrix4d dummy = Eigen::Matrix4d::Identity();
        dummy.topRightCorner<3, 1>() = target_position_in_odom;
        target_pose_pub_->publish(std::make_unique<nav_msgs::msg::Odometry>(
            to_odom_msg(dummy, now(), "world", "camera")));

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = now();
        tf.header.frame_id = "world";
        tf.child_frame_id = "map";
        tf.transform.translation.x = T(0, 3);
        tf.transform.translation.y = T(1, 3);
        tf.transform.translation.z = T(2, 3);
        const Eigen::Quaterniond q(Eigen::Matrix3d(T.topLeftCorner<3, 3>()));
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf_broadcaster_->sendTransform(tf);
    }

    static int closest_index(const Eigen::MatrixXd& paths, const Eigen::Vector3d& pos) {
        int best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (int i = 0; i < paths.rows(); ++i) {
            const double d = (paths.row(i).head<2>().transpose() - pos.head<2>()).norm();
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        return best;
    }

    // Port of lookahead_distance_m.
    static double lookahead_distance_m(double speed_cap_mps, double gain) {
        const double speed =
            std::isfinite(speed_cap_mps) ? speed_cap_mps * gain : kNoCapSpeedMps;
        return std::clamp(speed * kLookaheadS, kLookaheadMinM, kLookaheadMaxM);
    }

    // Port of select_target_position.
    Eigen::Vector3d select_target_position(const Eigen::MatrixXd& paths, int closest_idx,
                                           const Eigen::Vector3d& pos, double cap) {
        const double lookahead_m = lookahead_distance_m(cap, capture_speed_gain());
        double accumulated_distance = 0.0;
        Eigen::Vector3d start_point = pos;
        Eigen::Vector3d target_position = paths.row(paths.rows() - 1).transpose();
        for (int i = closest_idx; i < paths.rows() - 1; ++i) {
            accumulated_distance +=
                (paths.row(i).head<2>().transpose() - start_point.head<2>()).norm();
            if (accumulated_distance > lookahead_m) {
                target_position = paths.row(i).transpose();
                break;
            }
            start_point = paths.row(i).transpose();
        }
        return target_position;
    }

    static double capture_speed_gain() {
        const char* env = std::getenv("TINYNAV_CAPTURE_SPEED_GAIN");
        return env != nullptr ? std::atof(env) : 1.0;
    }

    // Port of poi_reached (arrival measured from the CAMERA pose; the tick
    // confirmation guards relocalization jumps).
    bool poi_reached(const Eigen::Vector3d& poi, const Eigen::Vector3d& pos) {
        const double arrive_m =
            poi_has_heading_.count(poi_index_) ? kArriveHeadingM : kArriveM;
        const bool inside = (poi.head<2>() - pos.head<2>()).norm() < arrive_m &&
                            std::abs(poi.z() - pos.z()) < 2.0;
        arrive_ticks_ = inside ? arrive_ticks_ + 1 : 0;
        return inside && arrive_ticks_ >= kArriveTicks;
    }

    // Port of update_leg_progress.
    std::pair<double, double> update_leg_progress(double remaining_length) {
        const rclcpp::Time now_t = now();  // wall-clock semantics in Python
        if (!leg_initial_length_.has_value()) {
            leg_initial_length_ = remaining_length;
            leg_start_time_ = now_t;
        }
        const double covered = *leg_initial_length_ - remaining_length;
        const double elapsed = (now_t - *leg_start_time_).seconds();
        if (covered > 0.1 && elapsed > 1.0) {
            speed_estimate_ = covered / elapsed;
        }
        const double initial = *leg_initial_length_;
        const double percent =
            initial > 0.0 ? std::clamp(covered / initial * 100.0, 0.0, 100.0) : 0.0;
        const double estimated_remaining_s =
            speed_estimate_.has_value() ? remaining_length / *speed_estimate_ : -1.0;
        return {percent, estimated_remaining_s};
    }

    void publish_nav_progress(double percent, double path_remaining_m, double path_total_m,
                              double estimated_remaining_s, bool arrived) {
        std_msgs::msg::String msg;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "{\"arrived\": %s, \"poi_index\": %d, \"percent\": %.1f, "
                      "\"path_remaining_m\": %.2f, \"path_total_m\": %.2f, "
                      "\"estimated_remaining_s\": %.1f}",
                      arrived ? "true" : "false", poi_index_, percent, path_remaining_m,
                      path_total_m, estimated_remaining_s);
        msg.data = buf;
        nav_progress_pub_->publish(std::make_unique<std_msgs::msg::String>(std::move(msg)));
    }

    void publish_global_plan(const Eigen::MatrixXd& paths_in_map) {
        nav_msgs::msg::Path path;
        path.header.stamp = now();
        path.header.frame_id = "map";
        for (int i = 0; i < paths_in_map.rows(); ++i) {
            geometry_msgs::msg::PoseStamped stamped;
            stamped.header = path.header;
            stamped.pose.position.x = paths_in_map(i, 0);
            stamped.pose.position.y = paths_in_map(i, 1);
            stamped.pose.position.z = paths_in_map(i, 2);
            stamped.pose.orientation.w = 1.0;
            path.poses.push_back(stamped);
        }
        global_plan_pub_->publish(std::make_unique<nav_msgs::msg::Path>(std::move(path)));
    }

    // Port of generate_nav_path_in_map.
    std::optional<Eigen::MatrixXd> generate_nav_path_in_map(const Eigen::Matrix4d& pose_in_map,
                                                            const Eigen::Vector3d& target_poi) {
        if (sdf_map_.shape()[0] == 0 || occupancy_map_.shape()[0] == 0) {
            RCLCPP_WARN(get_logger(),
                        "generate_nav_path_in_map: no occupancy/sdf map loaded (map format v2 pending)");
            return std::nullopt;
        }
        const Eigen::Vector3d origin = occupancy_map_origin_;
        const double resolution = occupancy_resolution_;
        // int() truncation toward zero, exactly as the Python casts.
        const Eigen::Vector3i start_idx =
            ((pose_in_map.topRightCorner<3, 1>() - origin) / resolution).cast<int>();
        const Eigen::Vector3i goal_idx = ((target_poi - origin) / resolution).cast<int>();
        const auto inside = [&](const Eigen::Vector3i& idx) {
            return 0 <= idx[0] && idx[0] < occupancy_map_.shape()[0] && 0 <= idx[1] &&
                   idx[1] < occupancy_map_.shape()[1] && 0 <= idx[2] &&
                   idx[2] < occupancy_map_.shape()[2];
        };
        if (!inside(start_idx) || !inside(goal_idx)) {
            return std::nullopt;
        }
        const mapping::GridCell start_cell{start_idx[0], start_idx[1], start_idx[2]};
        const mapping::GridCell goal_cell{goal_idx[0], goal_idx[1], goal_idx[2]};
        std::vector<mapping::GridCell> sdf_start_path =
            mapping::search_close_to_sdf_map(start_cell, sdf_map_, occupancy_map_, 0.2);
        std::vector<mapping::GridCell> sdf_goal_path =
            mapping::search_close_to_sdf_map(goal_cell, sdf_map_, occupancy_map_, 0.2);
        if (sdf_start_path.empty() || sdf_goal_path.empty()) {
            return std::nullopt;
        }
        const mapping::GridCell sdf_start = sdf_start_path.back();
        const mapping::GridCell sdf_goal = sdf_goal_path.back();
        std::vector<mapping::GridCell> path_sdf =
            mapping::search_within_sdf_map(sdf_start, sdf_goal, sdf_map_, occupancy_map_, resolution);
        if (path_sdf.empty()) {
            RCLCPP_WARN(get_logger(),
                        "search_within_sdf_map returned empty path: start_idx=(%d,%d,%d), goal_idx=(%d,%d,%d)",
                        sdf_start.x, sdf_start.y, sdf_start.z, sdf_goal.x, sdf_goal.y, sdf_goal.z);
        }
        std::vector<mapping::GridCell> path = sdf_start_path;
        path.insert(path.end(), path_sdf.begin(), path_sdf.end());
        path.insert(path.end(), sdf_goal_path.rbegin(), sdf_goal_path.rend());
        if (path.empty()) return std::nullopt;
        Eigen::MatrixXd converted(static_cast<int>(path.size()), 3);
        for (int i = 0; i < converted.rows(); ++i) {
            converted(i, 0) = path[static_cast<size_t>(i)].x * resolution + origin[0];
            converted(i, 1) = path[static_cast<size_t>(i)].y * resolution + origin[1];
            converted(i, 2) = path[static_cast<size_t>(i)].z * resolution + origin[2];
        }
        return converted;
    }

    // -------------------------------------------------------------- helpers
    static int64_t stamp_to_ns(const builtin_interfaces::msg::Time& stamp) {
        return static_cast<int64_t>(stamp.sec) * 1'000'000'000LL + static_cast<int64_t>(stamp.nanosec);
    }

    static Eigen::Matrix4d odom_to_T(const nav_msgs::msg::Odometry& msg) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        const auto& p = msg.pose.pose;
        T(0, 3) = p.position.x;
        T(1, 3) = p.position.y;
        T(2, 3) = p.position.z;
        const Eigen::Quaterniond quat(p.orientation.w, p.orientation.x, p.orientation.y,
                                      p.orientation.z);
        T.topLeftCorner<3, 3>() = quat.normalized().toRotationMatrix();
        return T;
    }

    static nav_msgs::msg::Odometry to_odom_msg(const Eigen::Matrix4d& T,
                                               const rclcpp::Time& stamp,
                                               const std::string& frame_id,
                                               const std::string& child_frame_id) {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = frame_id;
        odom.child_frame_id = child_frame_id;
        odom.pose.pose.position.x = T(0, 3);
        odom.pose.pose.position.y = T(1, 3);
        odom.pose.pose.position.z = T(2, 3);
        const Eigen::Quaterniond quat(Eigen::Matrix3d(T.topLeftCorner<3, 3>()));
        odom.pose.pose.orientation.x = quat.x();
        odom.pose.pose.orientation.y = quat.y();
        odom.pose.pose.orientation.z = quat.z();
        odom.pose.pose.orientation.w = quat.w();
        return odom;
    }

    // Minimal JSON reader for the POI payload {"<i>": {"position": [...],
    // "yaw_deg": x}} — the Python leans on json.loads.
    static bool parse_pois(const std::string& text,
                           std::map<int, Eigen::Vector3d>& pois,
                           std::set<int>& has_heading) {
        // Find each "<int>": { ... } entry by brace scanning (payload is small
        // and machine-generated; a full parser is overkill here).
        pois.clear();
        has_heading.clear();
        size_t i = 0;
        while ((i = text.find('"', i)) != std::string::npos) {
            const size_t key_end = text.find('"', i + 1);
            if (key_end == std::string::npos) return false;
            const std::string key = text.substr(i + 1, key_end - i - 1);
            if (key.find_first_not_of("0123456789") != std::string::npos) {
                i = key_end + 1;
                continue;
            }
            const int index = std::stoi(key);
            const size_t obj_open = text.find('{', key_end);
            if (obj_open == std::string::npos) return false;
            const size_t obj_close = text.find('}', obj_open);
            if (obj_close == std::string::npos) return false;
            const std::string obj = text.substr(obj_open, obj_close - obj_open + 1);
            const size_t pos_pos = obj.find("\"position\"");
            if (pos_pos == std::string::npos) return false;
            const size_t arr_open = obj.find('[', pos_pos);
            const size_t arr_close = obj.find(']', arr_open);
            if (arr_open == std::string::npos || arr_close == std::string::npos) return false;
            std::stringstream nums(obj.substr(arr_open + 1, arr_close - arr_open - 1));
            std::string token;
            std::vector<double> xyz;
            while (std::getline(nums, token, ',')) {
                try {
                    xyz.push_back(std::stod(token));
                } catch (const std::exception&) {
                    return false;
                }
            }
            if (xyz.size() != 3) return false;
            pois[index] = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
            if (obj.find("\"yaw_deg\"") != std::string::npos &&
                obj.find("\"yaw_deg\": null") == std::string::npos &&
                obj.find("\"yaw_deg\":null") == std::string::npos) {
                has_heading.insert(index);
            }
            i = obj_close + 1;
        }
        return true;
    }

    // --- state -------------------------------------------------------------
    std::unique_ptr<trt::SuperPointTRT> super_point_extractor_;
    std::unique_ptr<trt::LightGlueTRT> light_glue_matcher_;
    std::unique_ptr<trt::Dinov2TRT> dinov2_model_;
    trt::TrtOutputMap superpoint_results_;

    bool climb_prior_ = true;
    bool relocalization_enabled_ = false;

    // map state (the readable part of the map directory)
    mapping::OccupancyGrid occupancy_map_;
    mapping::SdfGrid sdf_map_;
    Eigen::Vector3d occupancy_map_origin_ = Eigen::Vector3d::Zero();
    double occupancy_resolution_ = 0.1;
    std::optional<mapping::PathSpeedIndex> speed_index_;
    std::optional<mapping::PathClimbIndex> climb_index_;

    // live keyframe state (nav_temp_db replacement): depth/features on disk
    // (LiveCapture, scratch-wiped per session), embeddings stay in RAM —
    // find_loop scans them on every keyframe and 3KB each is nothing.
    mapping::LiveCapture live_;
    std::unordered_map<int64_t, Eigen::VectorXd> embeddings_;
    std::pair<int64_t, Features> latest_keyframe_features_{-1, Features{}};

    std::map<int64_t, Eigen::Matrix4d> odom_;
    std::optional<int64_t> last_keyframe_timestamp_;
    cv::Mat last_keyframe_image_;
    std::map<int64_t, Eigen::Matrix4d> pose_graph_used_pose_;
    std::vector<std::tuple<int64_t, int64_t, Eigen::Matrix4d>> relative_pose_constraint_;
    std::map<int64_t, Eigen::Matrix4d> relocalization_poses_;
    std::map<int64_t, double> relocalization_pose_weights_;
    std::optional<Eigen::Matrix4d> T_from_map_to_odom_;

    std::optional<Eigen::Matrix3d> K_;
    double baseline_ = 0.0;
    std::optional<Eigen::Matrix3d> map_K_;  // the map's build-time intrinsics
    std::optional<Eigen::Matrix4d> latest_odom_pose_;

    // map format v2 index (poses + VLAD + per-keyframe features/depth)
    mapping::MapV2 map_index_;
    std::vector<int64_t> failed_relocalizations_;

    // reloc failure dump channel (see reloc_dump_failure)
    std::string reloc_dump_dir_;
    int reloc_dump_count_ = 0;
    bool reloc_dump_cap_logged_ = false;

    // POI / nav state
    std::map<int, Eigen::Vector3d> pois_;
    std::set<int> poi_has_heading_;
    int poi_index_ = -1;
    int arrive_ticks_ = 0;
    bool nav_completed_ = false;
    std::optional<double> leg_initial_length_;
    std::optional<rclcpp::Time> leg_start_time_;
    std::optional<double> speed_estimate_;
    std::optional<Eigen::MatrixXd> cached_nav_path_in_map_;
    int cached_nav_path_poi_index_ = -1;

    using Sync = message_filters::Synchronizer<message_filters::sync_policies::ExactTime<
        Image, nav_msgs::msg::Odometry, Image>>;
    message_filters::Subscriber<Image> depth_sub_;
    message_filters::Subscriber<Image> keyframe_image_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> keyframe_odom_sub_;
    std::shared_ptr<Sync> sync_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr continuous_odom_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pois_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_graph_trajectory_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr relocation_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr current_pose_in_map_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_cap_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr climb_region_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr on_stairs_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr localization_data_saved_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr poi_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr poi_change_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr nav_done_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr nav_progress_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr current_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_plan_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr target_pose_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr nav_target_timer_;
    rclcpp::TimerBase::SharedPtr map_prior_timer_;
};

// main.cpp's factory contract.
std::shared_ptr<rclcpp::Node> make_mapping(const rclcpp::NodeOptions& options) {
    return std::make_shared<MappingComponent>(options);
}

}  // namespace tinynav
