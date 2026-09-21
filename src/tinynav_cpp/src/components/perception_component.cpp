// Port of reference/tinynav/core/perception_node.py::PerceptionNode — v2:
// stereo depth + SuperPoint/LightGlue + PnP pose chaining over a keyframe
// window, the IMU bookkeeping and /slam/reset semantics, plus the python's
// window factor-graph refinement (CombinedImuFactor + smart stereo factors +
// LM) behind the tinynav::gtsam::Refine seam. Without the GTSAM build tree the
// component degrades to v1: keyframe poses are the chained PnP estimates.
//
// Deliberate divergences (per the migration plan / README known gaps):
//  - rclpy's InputAligner (IMU/stereo dispatch pairing) is not ported: IMU
//    messages are handled directly on their callback and the stereo worker
//    drains the same shared deque, so the consume-order contract the aligner
//    provided is preserved by the imu_measurements lock.
//  - The IMU preintegration is batched per window pair inside the Refine impl
//    (dt = stamp - previous stamp) instead of the python's per-frame drain
//    with its peeked-sample re-integration quirk (<10 ms per pair; see
//    RefineInput::imu).
//  - publish_selected stats keep the same /slam/data JSON keys that exist in
//    v1; num_tracks/num_factors/num_variables/errors come from the graph.
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/core.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>

#include "tinynav_cpp/core/imu.hpp"
#include "tinynav_cpp/core/math.hpp"
#include "tinynav_cpp/gtsam/refine.hpp"
#include "tinynav_cpp/trt/models.hpp"

namespace tinynav {

// The factor-graph seam (tinynav_cpp/gtsam/refine.hpp): the implementation
// refines the window poses in place (the Python writes result.atPose3(X(i))
// back into keyframe.pose / .velocity) and reports the errors for /slam/data.
using GtsamRefine = tinynav::gtsam::Refine;
using RefineInput = tinynav::gtsam::RefineInput;
using SmartObservation = tinynav::gtsam::SmartObservation;

class PerceptionComponent : public rclcpp::Node {
  public:
    explicit PerceptionComponent(const rclcpp::NodeOptions& options)
        : Node("perception_node", options) {
        std::string model_dir = "/tinynav/tinynav/models";
        declare_parameter<std::string>("model_dir", model_dir);
        model_dir = get_parameter("model_dir").as_string();

        superpoint_ = std::make_shared<trt::SuperPointTRT>(model_dir);
        light_glue_ = std::make_shared<trt::LightGlueTRT>(model_dir);
        stereo_engine_ = std::make_shared<trt::StereoEngineTRT>(model_dir);
        refine_ = tinynav::gtsam::make_gtsam_refine();
        if (refine_ != nullptr && refine_->available()) {
            RCLCPP_INFO(get_logger(), "GTSAM factor-graph refinement enabled.");
        } else {
            RCLCPP_WARN(get_logger(),
                        "GTSAM refine unavailable: keyframe poses stay the chained "
                        "PnP estimates (v1 behaviour)");
        }

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        rclcpp::QoS imu_qos(rclcpp::KeepLast(500));
        imu_qos.best_effort();
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/camera/camera/imu", imu_qos,
            [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { imu_callback(*msg); });
        camerainfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/infra2/camera_info", 10,
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { info_callback(*msg); });
        reset_sub_ = create_subscription<std_msgs::msg::Empty>(
            "/slam/reset", 10,
            [this](std_msgs::msg::Empty::ConstSharedPtr) {
                vio_reset_ = true;
                RCLCPP_WARN(get_logger(),
                            "VIO reset requested (/slam/reset): window clears on the next stereo frame");
            });
        // ApproximateTimeSynchronizer(queue_size=10, slop=0.02)
        left_sub_.subscribe(this, "/camera/camera/infra1/image_rect_raw");
        right_sub_.subscribe(this, "/camera/camera/infra2/image_rect_raw");
        sync_ = std::make_shared<Sync>(SyncPolicy(10), left_sub_, right_sub_);
        sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.02));
        sync_->registerCallback(std::bind(&PerceptionComponent::images_callback, this,
                                          std::placeholders::_1, std::placeholders::_2));

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/slam/odometry_visual", 10);
        slam_camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/slam/camera_info", 10);
        depth_pub_ = create_publisher<Image>("/slam/depth", 10);
        keyframe_pose_pub_ = create_publisher<nav_msgs::msg::Odometry>("/slam/keyframe_odom", 10);
        keyframe_image_pub_ = create_publisher<Image>("/slam/keyframe_image", 10);
        keyframe_depth_pub_ = create_publisher<Image>("/slam/keyframe_depth", 10);
        stats_pub_ = create_publisher<std_msgs::msg::String>("/slam/data", 10);

        worker_ = std::thread([this] { process_stereo_worker(); });
        RCLCPP_INFO(get_logger(), "PerceptionNode initialized.");
    }

    ~PerceptionComponent() override {
        stopped_ = true;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_closed_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

  private:
    using Image = sensor_msgs::msg::Image;

    // Port of the Keyframe dataclass. kpts/desc/mask cache the SuperPoint
    // extraction the python re-derives through its alru_cache.
    struct Keyframe {
        double timestamp = 0.0;
        cv::Mat image;
        cv::Mat disparity;
        cv::Mat depth;
        cv::Mat kpts;      // [1, N, 2] CV_32F
        cv::Mat desc;      // [1, N, D]
        cv::Mat mask;      // [1, N, 1]
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        int imu_measurement_count = 0;
        double latest_imu_timestamp = 0.0;
    };

    struct StereoPairMsg {
        std_msgs::msg::Header header;
        Image::ConstSharedPtr left_msg;
        Image::ConstSharedPtr right_msg;
    };

    static constexpr int kN = 5;             // keyframe window
    static constexpr int kM = 1000;          // python _M: UF index stride
    static constexpr int kMinFeatures = 20;
    static constexpr int kMinPnpInliers = 20;  // python: len(inlier_set) > 20
    static constexpr int kMinTrackObservations = 2;
    static constexpr double kKeyframeMinDistance = 0.1;      // m
    static constexpr double kKeyframeMinRotateDegree = 0.1;  // deg

    static double stamp2second(const builtin_interfaces::msg::Time& stamp) {
        const int64_t nano =
            static_cast<int64_t>(stamp.sec) * 1'000'000'000 + static_cast<int64_t>(stamp.nanosec);
        return static_cast<double>(nano) * 1e-9;
    }

    // Port of keyframe_check.
    static bool keyframe_check(const Eigen::Matrix4d& T_i, const Eigen::Matrix4d& T_j) {
        const Eigen::Matrix4d T_ij = T_i.inverse() * T_j;
        const double t_diff = T_ij.topRightCorner<3, 1>().norm();
        const double cos_theta = (T_ij.topLeftCorner<3, 3>().trace() - 1.0) / 2.0;
        const double r_diff =
            std::acos(std::clamp(cos_theta, -1.0, 1.0)) * 180.0 / M_PI;
        return t_diff > kKeyframeMinDistance || r_diff > kKeyframeMinRotateDegree;
    }

    // --- callbacks ---------------------------------------------------------
    // Port of _process_imu_msg (gravity alignment on the first 10 readings,
    // timestamp-jump warning, deque append). Runs on the executor thread.
    void imu_callback(const sensor_msgs::msg::Imu& imu_msg) {
        const double current_timestamp = stamp2second(imu_msg.header.stamp);
        std::vector<Eigen::Vector3d> accel_window;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            if (accel_readings_.size() >= 10 && !T_body_last_.has_value()) {
                accel_window = accel_readings_;
            } else if (accel_readings_.size() < 10) {
                accel_readings_.push_back(
                    Eigen::Vector3d(imu_msg.linear_acceleration.x,
                                    imu_msg.linear_acceleration.y,
                                    imu_msg.linear_acceleration.z));
            }
        }
        if (!accel_window.empty()) {
            Eigen::Vector3d gravity_cam = Eigen::Vector3d::Zero();
            for (const auto& a : accel_window) gravity_cam += a;
            gravity_cam /= static_cast<double>(accel_window.size());
            gravity_cam.normalize();
            const Eigen::Vector3d gravity_world(0.0, 0.0, 1.0);
            const Eigen::Matrix3d R_gravity_align =
                core::rot_from_two_vector(gravity_cam, gravity_world);
            const double initial_yaw =
                std::atan2(R_gravity_align(1, 0), R_gravity_align(0, 0)) + M_PI / 2.0;
            const double cos_yaw = std::cos(-initial_yaw), sin_yaw = std::sin(-initial_yaw);
            Eigen::Matrix3d R_zero_yaw;
            R_zero_yaw << cos_yaw, -sin_yaw, 0.0, sin_yaw, cos_yaw, 0.0, 0.0, 0.0, 1.0;
            Eigen::Matrix4d T_body_last = Eigen::Matrix4d::Identity();
            T_body_last.topLeftCorner<3, 3>() = R_zero_yaw * R_gravity_align;
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                T_body_last_ = T_body_last;
                // cold-start anchor: /slam/reset restores this, not the last pose
                T_align_ = T_body_last;
            }
            RCLCPP_INFO(get_logger(), "Initial yaw removed: %.2f deg",
                        initial_yaw * 180.0 / M_PI);
            RCLCPP_INFO(get_logger(), "Initial pose set from accelerometer data.");
        }

        if (imu_last_received_timestamp_.has_value() &&
            current_timestamp - *imu_last_received_timestamp_ > 0.1) {
            RCLCPP_WARN(get_logger(),
                        "IMU timestamp jump %.6f s is too large, it means the IMU is not working properly",
                        current_timestamp - *imu_last_received_timestamp_);
        }
        imu_last_received_timestamp_ = current_timestamp;
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_measurements_.push_back(ImuMeasurement{
            current_timestamp,
            Eigen::Vector3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z),
            Eigen::Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y,
                            imu_msg.angular_velocity.z)});
        // Batch preintegration needs every sample since the window's oldest
        // keyframe (python integrates on arrival, so its 1000-deep deque never
        // loses a span). 5 keyframes x 3 s keyframe-timeout at 100 Hz = 1500;
        // the post-refine prune keeps the deque at the window span anyway.
        if (imu_measurements_.size() > 4000) imu_measurements_.pop_front();
    }

    void info_callback(const sensor_msgs::msg::CameraInfo& msg) {
        if (K_.has_value()) return;
        Eigen::Matrix3d k;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) k(r, c) = msg.k[static_cast<size_t>(r * 3 + c)];
        }
        K_ = k;
        baseline_ = -msg.p[3] / k(0, 0);
        camera_info_msg_ = msg;
        RCLCPP_INFO(get_logger(), "Camera intrinsics and baseline received. Baseline: %.4fm",
                    baseline_);
        camerainfo_sub_.reset();
    }

    // Port of _aligned_stereo_callback: rate gate at 0.1333 s then a maxsize-1
    // queue that drops the oldest frame.
    void images_callback(Image::ConstSharedPtr left_msg, Image::ConstSharedPtr right_msg) {
        const double image_timestamp = stamp2second(left_msg->header.stamp);
        if (image_timestamp - last_processed_timestamp_ < 0.1333) {
            return;
        }
        last_processed_timestamp_ = image_timestamp;
        StereoPairMsg pair{left_msg->header, left_msg, right_msg};
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stereo_queue_.size() >= 1) {
                stereo_queue_.pop();
            }
            stereo_queue_.push(std::move(pair));
        }
        queue_cv_.notify_one();
    }

    // Port of _process_stereo_worker.
    void process_stereo_worker() {
        int consecutive_failures = 0;
        while (!stopped_) {
            StereoPairMsg pair;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !stereo_queue_.empty() || queue_closed_; });
                if (stopped_ && stereo_queue_.empty()) return;
                pair = std::move(stereo_queue_.front());
                stereo_queue_.pop();
            }
            const auto loop_start = std::chrono::steady_clock::now();
            try {
                process(*pair.left_msg, *pair.right_msg);
            } catch (const std::exception& exc) {
                // The window is rebuilt from scratch every frame; log and retry
                // with fresh geometry on the next frame.
                ++consecutive_failures;
                size_t imu_pending = 0;
                {
                    std::lock_guard<std::mutex> lock(imu_mutex_);
                    imu_pending = imu_measurements_.size();
                }
                RCLCPP_ERROR(get_logger(),
                             "VIO failed on frame at %.3f (%d consecutive): %s "
                             "keyframes=%zu imu_pending=%zu",
                             stamp2second(pair.header.stamp), consecutive_failures, exc.what(),
                             keyframe_queue_.size(), imu_pending);
                continue;
            }
            consecutive_failures = 0;
            const double loop_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          loop_start)
                    .count();
            publish_stats(loop_ms);
        }
    }

    // Port of process (async parts inlined; GTSAM solve behind GtsamRefine).
    void process(const Image& left_msg, const Image& right_msg) {
        if (!K_.has_value() || !T_body_last_.has_value()) {
            return;  // the Python's early stats frame
        }
        const cv::Mat left_img = cv_bridge::toCvCopy(left_msg, "mono8")->image;
        const cv::Mat right_img = cv_bridge::toCvCopy(right_msg, "mono8")->image;
        const double current_timestamp = stamp2second(left_msg.header.stamp);

        if (vio_reset_.exchange(false)) {
            // drop the poisoned window; the next branch re-anchors on this frame
            const size_t n = keyframe_queue_.size();
            keyframe_queue_.clear();
            T_body_last_ = T_align_.value_or(Eigen::Matrix4d::Identity());
            V_last_ = Eigen::Vector3d::Zero();
            RCLCPP_WARN(get_logger(),
                        "VIO backend reset: cleared %zu keyframes, re-anchoring on this frame", n);
        }

        cv::Mat disparity, depth;
        if (keyframe_queue_.empty()) {  // first frame
            if (!stereo_infer(left_img, right_img, disparity, depth)) {
                return;
            }
            Keyframe kf;
            kf.timestamp = current_timestamp;
            kf.image = left_img;
            kf.disparity = disparity;
            kf.depth = depth;
            kf.pose = *T_body_last_;
            kf.velocity = Eigen::Vector3d::Zero();
            kf.latest_imu_timestamp = current_timestamp;
            // cache the SuperPoint extraction the window association reuses
            if (!superpoint_->infer(left_img, results_["kf"])) {
                return;  // engine unavailable: keep waiting for a frame we can anchor
            }
            extract(results_["kf"], kf.kpts, kf.desc, kf.mask);
            keyframe_queue_.push_back(std::move(kf));
            return;
        }

        if (!stereo_infer(left_img, right_img, disparity, depth)) {
            return;
        }
        const Keyframe& kf_prev = keyframe_queue_.back();
        cv::Mat curr_kpts, curr_desc, curr_mask;
        if (!superpoint_->infer(left_img, results_["curr"])) {
            return;  // engines unavailable: degrade, keep the window
        }
        extract(results_["curr"], curr_kpts, curr_desc, curr_mask);

        // PnP chain match: the stored kf_prev extraction vs the current frame.
        cv::Mat match_indices;
        if (!light_glue_infer(kf_prev.kpts, kf_prev.desc, kf_prev.mask, curr_kpts,
                              curr_desc, curr_mask,
                              cv::Size(kf_prev.image.cols, kf_prev.image.rows),
                              cv::Size(left_img.cols, left_img.rows), match_indices)) {
            return;
        }

        // Snapshot the IMU samples up to the frame timestamp without popping
        // (the graph re-slices them per window pair; the deque is pruned to the
        // window span after the refine).
        std::vector<core::ImuSample> imu_upto_now;
        int imu_count = 0;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            for (const ImuMeasurement& m : imu_measurements_) {
                if (m.timestamp > current_timestamp) break;
                imu_upto_now.push_back(
                    core::ImuSample{m.timestamp, m.gyro, m.accel});
                ++imu_count;
            }
        }

        // PnP between the last keyframe and the current frame.
        const Eigen::MatrixX2d prev_keypoints = to_eigen_kpts(kf_prev.kpts);
        const Eigen::MatrixX2d current_keypoints = to_eigen_kpts(curr_kpts);
        const std::vector<int> match_indices_vec = to_index_vec(match_indices);
        std::vector<int> idx_valid;
        std::vector<Eigen::Vector2d> kpt_pre, kpt_cur;
        for (size_t i = 0; i < match_indices_vec.size(); ++i) {
            if (match_indices_vec[i] != -1) {
                kpt_pre.push_back(prev_keypoints.row(static_cast<int>(i)));
                kpt_cur.push_back(current_keypoints.row(match_indices_vec[i]));
                idx_valid.push_back(static_cast<int>(i));
            }
        }
        Eigen::MatrixXd depth_eigen = cv_mat_to_eigen(depth);
        const core::EstimatePoseResult est =
            core::estimate_pose(to_matrix(kpt_pre), to_matrix(kpt_cur), depth_eigen, *K_, idx_valid);
        if (!est.success) {
            // diagnostics: how deep did the chain get before PnP refused?
            int z_ok = 0;
            for (size_t i = 0; i < kpt_cur.size(); ++i) {
                const int u = static_cast<int>(kpt_cur[i][0]);
                const int v = static_cast<int>(kpt_cur[i][1]);
                if (v >= 0 && v < depth.rows && u >= 0 && u < depth.cols) {
                    const float z = depth.at<float>(v, u);
                    if (z > 0.1f && z < 10.f) ++z_ok;
                }
            }
            float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
            for (size_t i = 0; i < kpt_cur.size(); ++i) {
                umin = std::min(umin, static_cast<float>(kpt_cur[i][0]));
                umax = std::max(umax, static_cast<float>(kpt_cur[i][0]));
                vmin = std::min(vmin, static_cast<float>(kpt_cur[i][1]));
                vmax = std::max(vmax, static_cast<float>(kpt_cur[i][1]));
            }
            int depth_valid = 0;
            for (int i = 0; i < depth.rows; i += 5) {
                for (int j = 0; j < depth.cols; j += 5) {
                    const float z = depth.at<float>(i, j);
                    if (z > 0.1f && z < 10.f) ++depth_valid;
                }
            }
            RCLCPP_ERROR(get_logger(),
                         "estimate_pose failed: matches=%zu u=[%.1f,%.1f] v=[%.1f,%.1f] "
                         "depth_dims=%dx%d depth_valid=%d z_ok=%d",
                         kpt_pre.size(), umin, umax, vmin, vmax, depth.rows, depth.cols,
                         depth_valid, z_ok);
            throw std::runtime_error("estimate_pose failed on this frame");
        }
        const Eigen::Matrix4d T_kf_curr = est.pose;

        // for new frame, we first add it as keyframe, if not, we pop it later
        Keyframe kf;
        kf.timestamp = current_timestamp;
        kf.image = left_img;
        kf.disparity = disparity;
        kf.depth = depth;
        kf.kpts = curr_kpts;
        kf.desc = curr_desc;
        kf.mask = curr_mask;
        kf.pose = keyframe_queue_.back().pose * T_kf_curr;
        kf.velocity = keyframe_queue_.back().velocity;
        kf.imu_measurement_count = imu_count;
        kf.latest_imu_timestamp = current_timestamp;
        keyframe_queue_.push_back(std::move(kf));
        if (static_cast<int>(keyframe_queue_.size()) > kN) {
            keyframe_queue_.erase(keyframe_queue_.begin());
        }

        // [ISAM Processing] — the window factor graph, rebuilt from scratch
        // every frame (port of the python block with the same log name).
        if (refine_ != nullptr && refine_->available()) {
            // Window data association (python lines 511-580): LightGlue per
            // consecutive window pair, PnP-inlier filter, UF tracks.
            const int n = static_cast<int>(keyframe_queue_.size());
            std::vector<std::vector<SmartObservation>> tracks;
            std::vector<int> velocity_prior_pose_idx;
            if (n >= 2) {
                core::UnionFind uf(n * kM);
                for (int i = 0; i + 1 < n; ++i) {
                    const Keyframe& kfi = keyframe_queue_[static_cast<size_t>(i)];
                    const Keyframe& kfj = keyframe_queue_[static_cast<size_t>(i) + 1];
                    cv::Mat pair_indices;
                    if (!light_glue_infer(kfi.kpts, kfi.desc, kfi.mask, kfj.kpts,
                                          kfj.desc, kfj.mask,
                                          cv::Size(kfi.image.cols, kfi.image.rows),
                                          cv::Size(kfj.image.cols, kfj.image.rows),
                                          pair_indices)) {
                        return;
                    }
                    const Eigen::MatrixX2d kpts_pre = to_eigen_kpts(kfi.kpts);
                    const Eigen::MatrixX2d kpts_cur = to_eigen_kpts(kfj.kpts);
                    const std::vector<int> mi = to_index_vec(pair_indices);
                    std::vector<int> idx_valid;
                    std::vector<Eigen::Vector2d> kpt_pre, kpt_cur;
                    for (size_t k = 0; k < mi.size(); ++k) {
                        if (mi[k] != -1) {
                            kpt_pre.push_back(kpts_pre.row(static_cast<int>(k)));
                            kpt_cur.push_back(kpts_cur.row(mi[k]));
                            idx_valid.push_back(static_cast<int>(k));
                        }
                    }
                    // python filters against the CURRENT keyframe's depth.
                    const core::EstimatePoseResult est = core::estimate_pose(
                        to_matrix(kpt_pre), to_matrix(kpt_cur),
                        cv_mat_to_eigen(kfj.depth), *K_, idx_valid);
                    const std::vector<int>& inliers = est.inlier_idx_original;
                    const std::set<int> inlier_set(inliers.begin(), inliers.end());
                    std::vector<int> filtered = mi;
                    if (static_cast<int>(inlier_set.size()) > kMinPnpInliers) {
                        for (size_t k = 0; k < filtered.size(); ++k) {
                            if (filtered[k] != -1 &&
                                inlier_set.count(static_cast<int>(k)) == 0) {
                                filtered[k] = -1;
                            }
                        }
                    } else {
                        RCLCPP_WARN(get_logger(),
                                    "match cnt: %zu is too small, %zu inliers. "
                                    "enable velocity constraint",
                                    kpt_pre.size(), inlier_set.size());
                        std::fill(filtered.begin(), filtered.end(), -1);
                        velocity_prior_pose_idx.push_back(i);
                    }
                    int count = 0;
                    for (size_t k = 0; k < filtered.size(); ++k) {
                        if (filtered[k] != -1) {
                            uf.unite(i * kM + static_cast<int>(k),
                                     (i + 1) * kM + filtered[k]);
                            ++count;
                        }
                    }
                }
                // tracks → smart-factor observations with the disparity check
                const std::vector<std::vector<int>> parts =
                    core::uf_all_sets_list(uf, kMinTrackObservations);
                std::vector<Eigen::MatrixX2d> window_kpts(n);
                for (int i = 0; i < n; ++i) {
                    window_kpts[static_cast<size_t>(i)] =
                        to_eigen_kpts(keyframe_queue_[static_cast<size_t>(i)].kpts);
                }
                for (const std::vector<int>& landmark : parts) {
                    bool disparity_valid = true;
                    std::vector<SmartObservation> observations;
                    for (const int projection : landmark) {
                        const int pose_idx = projection / kM;
                        const int feature_idx = projection % kM;
                        const Keyframe& kf = keyframe_queue_[static_cast<size_t>(pose_idx)];
                        const double u = window_kpts[static_cast<size_t>(pose_idx)](
                            feature_idx, 0);
                        const double v = window_kpts[static_cast<size_t>(pose_idx)](
                            feature_idx, 1);
                        const int ui = static_cast<int>(u), vi = static_cast<int>(v);
                        if (vi < 0 || vi >= kf.disparity.rows || ui < 0 ||
                            ui >= kf.disparity.cols ||
                            kf.disparity.at<float>(vi, ui) < 0.1) {
                            disparity_valid = false;
                            break;
                        }
                        observations.push_back(SmartObservation{
                            pose_idx, u, u - kf.disparity.at<float>(vi, ui), v});
                    }
                    if (!disparity_valid ||
                        static_cast<int>(observations.size()) < kMinTrackObservations) {
                        continue;
                    }
                    tracks.push_back(std::move(observations));
                }
            }

            RefineInput input;
            input.keyframe_timestamps.reserve(keyframe_queue_.size());
            for (const Keyframe& k : keyframe_queue_) {
                input.keyframe_timestamps.push_back(k.timestamp);
            }
            input.imu = std::move(imu_upto_now);
            input.tracks = std::move(tracks);
            input.velocity_prior_pose_idx = std::move(velocity_prior_pose_idx);
            input.K = *K_;
            input.baseline = baseline_;

            std::vector<Eigen::Matrix4d> poses;
            std::vector<Eigen::Vector3d> velocities;
            for (const Keyframe& k : keyframe_queue_) {
                poses.push_back(k.pose);
                velocities.push_back(k.velocity);
            }
            const GtsamRefine::Metrics metrics = refine_->refine(poses, velocities, input);
            for (size_t i = 0; i < keyframe_queue_.size(); ++i) {
                keyframe_queue_[i].pose = poses[i];
                keyframe_queue_[i].velocity = velocities[i];
            }
            last_metrics_ = metrics;
            RCLCPP_INFO(get_logger(),
                        "ISAM optimization done with %d factors and %d variables: "
                        "initial %.4f final %.4f",
                        metrics.num_factors, metrics.num_variables,
                        metrics.initial_error, metrics.final_error);

            // prune the IMU deque to what the remaining window can still need
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                const double window_start = keyframe_queue_.front().timestamp;
                while (!imu_measurements_.empty() &&
                       imu_measurements_.front().timestamp <= window_start) {
                    imu_measurements_.pop_front();
                }
            }
        }

        // publish depth image and camera info for the depth topic (DepthCloud)
        auto depth_out = cv_bridge::CvImage(left_msg.header, "32FC1", depth).toImageMsg();
        depth_out->header.frame_id = "camera";
        auto info_out = std::make_unique<sensor_msgs::msg::CameraInfo>(camera_info_msg_);
        info_out->header.stamp = left_msg.header.stamp;
        info_out->header.frame_id = "camera";
        slam_camera_info_pub_->publish(std::move(info_out));
        depth_pub_->publish(*depth_out);

        // publish odometry + TF (np2msg / np2tf)
        T_body_last_ = keyframe_queue_.back().pose;
        V_last_ = keyframe_queue_.back().velocity;
        odom_pub_->publish(
            std::make_unique<nav_msgs::msg::Odometry>(to_odom_msg(
                *T_body_last_, left_msg.header.stamp, "world", "camera", *V_last_)));
        geometry_msgs::msg::TransformStamped tf;
        tf.header = to_odom_msg(*T_body_last_, left_msg.header.stamp, "world", "camera",
                                *V_last_)
                        .header;
        tf.child_frame_id = "camera";
        tf.transform.translation.x = (*T_body_last_)(0, 3);
        tf.transform.translation.y = (*T_body_last_)(1, 3);
        tf.transform.translation.z = (*T_body_last_)(2, 3);
        const Eigen::Quaterniond q(
            Eigen::Matrix3d(T_body_last_->topLeftCorner<3, 3>()));
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf_broadcaster_->sendTransform(tf);

        // keyframe decision
        const Keyframe& last_keyframe = keyframe_queue_[keyframe_queue_.size() - 2];
        const Keyframe& current_keyframe = keyframe_queue_.back();
        if (keyframe_check(last_keyframe.pose, current_keyframe.pose) ||
            current_keyframe.timestamp - last_keyframe.timestamp > 3.0) {
            keyframe_pose_pub_->publish(std::make_unique<nav_msgs::msg::Odometry>(
                to_odom_msg(current_keyframe.pose, left_msg.header.stamp, "world", "camera",
                            current_keyframe.velocity)));
            keyframe_image_pub_->publish(std::make_unique<Image>(left_msg));
            auto kf_depth = cv_bridge::CvImage(left_msg.header, "32FC1", depth).toImageMsg();
            kf_depth->header.frame_id = "camera";
            keyframe_depth_pub_->publish(*kf_depth);
        } else {
            keyframe_queue_.pop_back();
        }
    }

    bool stereo_infer(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity,
                      cv::Mat& depth) {
        if (!stereo_engine_->infer(left, right, baseline_, (*K_)(0, 0), disparity, depth)) {
            // Engine load failure degrades: without depth the chain cannot run.
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "stereo engine unavailable; perception idle");
            return false;
        }
        return true;
    }

    bool light_glue_infer(const cv::Mat& kpts0, const cv::Mat& desc0, const cv::Mat& mask0,
                          const cv::Mat& kpts1, const cv::Mat& desc1, const cv::Mat& mask1,
                          const cv::Size& shape0, const cv::Size& shape1,
                          cv::Mat& match_indices) {
        const std::array<int64_t, 2> s0{shape0.width, shape0.height};
        const std::array<int64_t, 2> s1{shape1.width, shape1.height};
        trt::TrtOutputMap out;
        if (!light_glue_->infer(kpts0, kpts1, desc0, desc1, mask0, mask1, s0, s1, out)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "lightglue engine unavailable; perception idle");
            return false;
        }
        match_indices = out.at("match_indices");
        return true;
    }

    static void extract(const trt::TrtOutputMap& result, cv::Mat& kpts, cv::Mat& descps,
                        cv::Mat& mask) {
        kpts = result.at("kpts");
        descps = result.at("descps");
        const auto it = result.find("mask");
        if (it != result.end()) mask = it->second;
    }

    void publish_stats(double loop_ms) {
        std_msgs::msg::String stats;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "{\"process_cnt\": %zu, \"num_keyframes\": %zu, "
                      "\"num_tracks\": %d, \"num_factors\": %d, \"num_variables\": %d, "
                      "\"initial_error\": %.4f, \"final_error\": %.4f, \"loop_ms\": %.1f}",
                      process_cnt_, keyframe_queue_.size(), last_metrics_.num_tracks,
                      last_metrics_.num_factors, last_metrics_.num_variables,
                      last_metrics_.initial_error, last_metrics_.final_error, loop_ms);
        stats.data = buf;
        stats_pub_->publish(std::make_unique<std_msgs::msg::String>(std::move(stats)));
    }

    // --- helpers -----------------------------------------------------------
    static Eigen::MatrixX2d to_matrix(const std::vector<Eigen::Vector2d>& pts) {
        Eigen::MatrixX2d out(static_cast<int>(pts.size()), 2);
        for (int i = 0; i < out.rows(); ++i) out.row(i) = pts[static_cast<size_t>(i)];
        return out;
    }

    static Eigen::MatrixX2d to_eigen_kpts(const cv::Mat& kpts) {
        // kpts: [1, 2, N] or [1, N, 2] CV_32F (see SuperPointTRT.infer contract)
        Eigen::MatrixX2d out;
        if (kpts.empty()) return out;
        // [1,2,N] keeps N in size[2]; [1,N,2] keeps it in size[1] — the engine
        // emits [1,512,2], so picking size[2] blindly read the coordinate axis.
        const int n = kpts.dims == 3 && kpts.size[1] == 2 ? kpts.size[2] : kpts.size[1];
        out.resize(n, 2);
        for (int i = 0; i < n; ++i) {
            if (kpts.dims == 3 && kpts.size[1] == 2) {
                out(i, 0) = kpts.at<float>(0, 0, i);
                out(i, 1) = kpts.at<float>(0, 1, i);
            } else {
                out(i, 0) = kpts.at<float>(0, i, 0);
                out(i, 1) = kpts.at<float>(0, i, 1);
            }
        }
        return out;
    }

    static std::vector<int> to_index_vec(const cv::Mat& match_indices) {
        // kINT64 outputs arrive as CV_64F (see trt_engine.hpp); guard CV_32S too.
        std::vector<int> out;
        if (match_indices.empty()) return out;
        const int n = match_indices.size[match_indices.dims - 1];
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (match_indices.type() == CV_64F) {
                out.push_back(static_cast<int>(match_indices.at<double>(0, i)));
            } else if (match_indices.type() == CV_32S) {
                out.push_back(match_indices.at<int32_t>(0, i));
            } else {
                out.push_back(static_cast<int>(match_indices.at<float>(0, i)));
            }
        }
        return out;
    }

    static Eigen::MatrixXd cv_mat_to_eigen(const cv::Mat& depth) {
        Eigen::MatrixXd out(depth.rows, depth.cols);
        for (int r = 0; r < depth.rows; ++r) {
            for (int c = 0; c < depth.cols; ++c) {
                out(r, c) = depth.at<float>(r, c);
            }
        }
        return out;
    }

    static nav_msgs::msg::Odometry to_odom_msg(const Eigen::Matrix4d& T,
                                               const builtin_interfaces::msg::Time& stamp,
                                               const std::string& frame_id,
                                               const std::string& child_frame_id,
                                               const Eigen::Vector3d& velocity) {
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
        odom.twist.twist.linear.x = velocity[0];
        odom.twist.twist.linear.y = velocity[1];
        odom.twist.twist.linear.z = velocity[2];
        return odom;
    }

    // --- state -------------------------------------------------------------
    std::shared_ptr<trt::SuperPointTRT> superpoint_;
    std::shared_ptr<trt::LightGlueTRT> light_glue_;
    std::shared_ptr<trt::StereoEngineTRT> stereo_engine_;
    std::shared_ptr<GtsamRefine> refine_;  // v1: unset; lands with the GTSAM layer
    GtsamRefine::Metrics last_metrics_;
    std::unordered_map<std::string, trt::TrtOutputMap> results_;

    std::optional<Eigen::Matrix3d> K_;
    double baseline_ = 0.0;
    sensor_msgs::msg::CameraInfo camera_info_msg_;

    std::optional<Eigen::Matrix4d> T_body_last_;
    std::optional<Eigen::Matrix4d> T_align_;  // cold-start anchor
    std::optional<Eigen::Vector3d> V_last_;

    // IMU bookkeeping (guarded by imu_mutex_, shared with the worker)
    std::mutex imu_mutex_;
    std::vector<Eigen::Vector3d> accel_readings_;
    std::optional<double> imu_last_received_timestamp_;
    struct ImuMeasurement {
        double timestamp;
        Eigen::Vector3d accel;
        Eigen::Vector3d gyro;
    };
    std::deque<ImuMeasurement> imu_measurements_;

    std::atomic<bool> vio_reset_{false};
    double last_processed_timestamp_ = 0.0;
    std::atomic<bool> stopped_{false};
    std::queue<StereoPairMsg> stereo_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool queue_closed_ = false;
    std::thread worker_;
    std::vector<Keyframe> keyframe_queue_;
    size_t process_cnt_ = 0;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;
    message_filters::Subscriber<Image> left_sub_;
    message_filters::Subscriber<Image> right_sub_;
    std::shared_ptr<Sync> sync_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camerainfo_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr slam_camera_info_pub_;
    rclcpp::Publisher<Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr keyframe_pose_pub_;
    rclcpp::Publisher<Image>::SharedPtr keyframe_image_pub_;
    rclcpp::Publisher<Image>::SharedPtr keyframe_depth_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

// main.cpp's factory contract.
std::shared_ptr<rclcpp::Node> make_perception(const rclcpp::NodeOptions& options) {
    return std::make_shared<PerceptionComponent>(options);
}

}  // namespace tinynav
