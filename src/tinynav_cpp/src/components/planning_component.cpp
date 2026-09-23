// Port of reference/tinynav/core/planning_node.py::PlanningNode — the ROS
// shell; every kernel call goes through planning/planning.hpp (and the
// scipy replacements in planning/edt.hpp).
//
// Structure notes vs the Python:
//  - The declared parameters mirror __init__'s declare_parameter set and are
//    re-readable at runtime via on_set_parameters (the wave-2 contract).
//  - logsetup's every(1.0) becomes a last-status-stamp check on the node clock.
//  - Publishers/topics/QoS and the sync_callback flow follow the Python
//    section by section; see the per-block comments.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
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
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <opencv2/imgproc.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/point32.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32.hpp>

#include "tinynav_cpp/core/math.hpp"
#include "tinynav_cpp/core/robot_specs.hpp"
#include "tinynav_cpp/planning/planning.hpp"

namespace tinynav {

using sensor_msgs::msg::Image;
using message_filters::Synchronizer;
using message_filters::sync_policies::ExactTime;

class PlanningComponent : public rclcpp::Node {
  public:
    explicit PlanningComponent(const rclcpp::NodeOptions& options)
        : Node("planning_node", options) {

        path_pub_ = create_publisher<nav_msgs::msg::Path>("/planning/trajectory_path", 10);
        height_map_pub_ = create_publisher<Image>("/planning/height_map", 10);
        obstacle_mask_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/planning/obstacle_mask", 10);
        footprint_pub_ = create_publisher<sensor_msgs::msg::PointCloud>("/planning/footprint", 10);
        occupancy_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/planning/occupied_voxels", 10);
        occupancy_cloud_esdf_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/planning/occupied_voxels_with_esdf", 10);
        occupancy_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/planning/occupancy_grid", 10);

        // latest_depth_only / poses_covering_one_depth_frame QoS, then an exact
        // TimeSynchronizer(10) on (depth, odom).
        rclcpp::QoS latest_depth_only(rclcpp::KeepLast(1));
        latest_depth_only.reliable();
        rclcpp::QoS poses_covering_one_depth_frame(rclcpp::KeepLast(10));
        poses_covering_one_depth_frame.reliable();
        depth_sub_.subscribe(this, "/slam/depth", latest_depth_only.get_rmw_qos_profile());
        pose_sub_.subscribe(this, "/slam/odometry_visual",
                            poses_covering_one_depth_frame.get_rmw_qos_profile());
        sync_ = std::make_shared<Sync>(10, depth_sub_, pose_sub_);
        sync_->registerCallback(std::bind(&PlanningComponent::sync_callback, this,
                                          std::placeholders::_1, std::placeholders::_2));
        camerainfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/infra2/camera_info", 10,
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { info_callback(*msg); });

        declare_parameters();
        read_parameters();

        // Derive the grid z extent / offset from the obstacle band, as __init__ does.
        const int z_layers = static_cast<int>(std::round(
            (obstacle_config_.robot_z_top - obstacle_config_.robot_z_bottom) / resolution_));
        grid_shape_ = Eigen::Vector3i(100, 100, z_layers);
        z_grid_drop_ = -(obstacle_config_.robot_z_top + obstacle_config_.robot_z_bottom) / 2.0;
        origin_ = grid_shape_.cast<double>() * resolution_ / -2.0;
        free_space_esdf_ = std::hypot(static_cast<double>(grid_shape_[0]),
                                      static_cast<double>(grid_shape_[1])) *
                           resolution_;
        occupancy_grid_ = planning::OccupancyGrid3D(grid_shape_[0], grid_shape_[1], grid_shape_[2]);

        target_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/control/target_pose", 10,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { target_pose_callback(*msg); });
        poi_change_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/mapping/poi_change", 10,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr) {
                target_pose_ = std::nullopt;
                global_route_map_xy_ = Eigen::MatrixX2d();  // the cached route led to the old target
                has_cached_route_ = false;
            });
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        // /tf_static needs TRANSIENT_LOCAL, which rclcpp forbids under
        // intra-process comms — so the listener lives on a dedicated
        // non-intra-process sub-node and spins its own thread (the Python node
        // had no such restriction; rclpy has no intra-process feature).
        tf_node_ = std::make_shared<rclcpp::Node>(
            "planning_tf", rclcpp::NodeOptions().use_intra_process_comms(false));
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, tf_node_, true);
        global_route_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/mapping/global_plan", 1,
            [this](nav_msgs::msg::Path::ConstSharedPtr msg) { on_global_route(*msg); });

        climb_region_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
            "/planning/climb_region", 10,
            [this](sensor_msgs::msg::PointCloud::ConstSharedPtr msg) { climb_region_callback(*msg); });
        speed_cap_sub_ = create_subscription<std_msgs::msg::Float32>(
            "/planning/speed_cap", 10,
            [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
                speed_cap_ = msg->data;
                speed_cap_stamp_ns_ = now().nanoseconds();
            });

        RCLCPP_INFO(get_logger(),
                    "Robot: %s (%s %.2fx%.2fm, cam=(%.2f,%.2f), ctrl=(%.2f,%.2f), "
                    "safety_r=%.2fm, z_band=[%.2f, %.2f]m)",
                    robot_.name.c_str(), robot_.shape.c_str(), robot_.length, robot_.width,
                    robot_.camera_x, robot_.camera_y, robot_.control_x, robot_.control_y,
                    robot_.safety_radius, obstacle_config_.robot_z_bottom,
                    obstacle_config_.robot_z_top);
    }

  private:
    // --- parameters --------------------------------------------------------
    // planning_node.py's declare_parameter set (15 dynamic knobs).
    void declare_parameters() {
        declare_parameter("min_wall_span_m", robot_.obstacle.min_wall_span_m);
        declare_parameter("vx_max", 0.6);
        declare_parameter("vx_hard_max", 1.0);
        declare_parameter("vx_min", 0.2);
        declare_parameter("clear_c0_m", 0.35);
        declare_parameter("clear_open_m", 1.0);
        declare_parameter("clear_scan_m", 2.0);
        declare_parameter("t_react_s", 0.2);
        declare_parameter("traj_max_len_m", 2.5);
        declare_parameter("traj_max_lat_acc", 0.5);
        declare_parameter("climb_region_radius_m", 0.75);
        declare_parameter("climb_region_ttl_s", 3.0);
        declare_parameter("climb_min_wall_span_m", 0.2);
        declare_parameter("capture_speed_gain", capture_speed_gain_env());
        declare_parameter("speed_cap_ttl_s", 2.0);

        // Live-update the cached values on a runtime reconfigure.
        param_cb_ = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter>& params) {
                rcl_interfaces::msg::SetParametersResult result;
                result.successful = true;
                for (const auto& p : params) {
                    if (p.get_name() == "min_wall_span_m") {
                        obstacle_config_.min_wall_span_m = p.as_double();
                    } else if (p.get_name() == "vx_max") {
                        vx_max_ = p.as_double();
                    } else if (p.get_name() == "vx_hard_max") {
                        vx_hard_max_ = p.as_double();
                    } else if (p.get_name() == "vx_min") {
                        vx_min_ = p.as_double();
                    } else if (p.get_name() == "clear_c0_m") {
                        clear_c0_m_ = p.as_double();
                    } else if (p.get_name() == "clear_open_m") {
                        clear_open_m_ = p.as_double();
                    } else if (p.get_name() == "clear_scan_m") {
                        clear_scan_m_ = p.as_double();
                    } else if (p.get_name() == "t_react_s") {
                        t_react_s_ = p.as_double();
                    } else if (p.get_name() == "traj_max_len_m") {
                        traj_max_len_m_ = p.as_double();
                    } else if (p.get_name() == "traj_max_lat_acc") {
                        traj_max_lat_acc_ = p.as_double();
                    } else if (p.get_name() == "climb_region_radius_m") {
                        climb_region_cells_ =
                            static_cast<int>(std::round(p.as_double() / resolution_));
                    } else if (p.get_name() == "climb_region_ttl_s") {
                        climb_region_ttl_ns_ =
                            static_cast<int64_t>(p.as_double() * 1e9);
                    } else if (p.get_name() == "climb_min_wall_span_m") {
                        climb_min_wall_span_m_ = p.as_double();
                    } else if (p.get_name() == "capture_speed_gain") {
                        capture_speed_gain_ = p.as_double();
                    } else if (p.get_name() == "speed_cap_ttl_s") {
                        speed_cap_ttl_ns_ = static_cast<int64_t>(p.as_double() * 1e9);
                    }
                }
                return result;
            });
    }

    void read_parameters() {
        obstacle_config_ = robot_.obstacle;
        obstacle_config_.min_wall_span_m = get_parameter("min_wall_span_m").as_double();
        vx_max_ = get_parameter("vx_max").as_double();
        vx_hard_max_ = get_parameter("vx_hard_max").as_double();
        vx_min_ = get_parameter("vx_min").as_double();
        clear_c0_m_ = get_parameter("clear_c0_m").as_double();
        clear_open_m_ = get_parameter("clear_open_m").as_double();
        clear_scan_m_ = get_parameter("clear_scan_m").as_double();
        t_react_s_ = get_parameter("t_react_s").as_double();
        traj_max_len_m_ = get_parameter("traj_max_len_m").as_double();
        traj_max_lat_acc_ = get_parameter("traj_max_lat_acc").as_double();
        climb_region_cells_ = static_cast<int>(std::round(
            get_parameter("climb_region_radius_m").as_double() / resolution_));
        climb_region_ttl_ns_ = static_cast<int64_t>(
            get_parameter("climb_region_ttl_s").as_double() * 1e9);
        climb_min_wall_span_m_ = get_parameter("climb_min_wall_span_m").as_double();
        capture_speed_gain_ = get_parameter("capture_speed_gain").as_double();
        speed_cap_ttl_ns_ = static_cast<int64_t>(
            get_parameter("speed_cap_ttl_s").as_double() * 1e9);
    }

    // path_speed.py::CAPTURE_SPEED_GAIN reads TINYNAV_CAPTURE_SPEED_GAIN at import.
    static double capture_speed_gain_env() {
        const char* env = std::getenv("TINYNAV_CAPTURE_SPEED_GAIN");
        return env != nullptr ? std::atof(env) : 1.0;
    }

    // --- callbacks ---------------------------------------------------------
    void climb_region_callback(const sensor_msgs::msg::PointCloud& msg) {
        // An empty cloud is a real answer ("no region here"); only the stamp
        // decides freshness.
        climb_points_.resize(msg.points.size(), 2);
        for (size_t i = 0; i < msg.points.size(); ++i) {
            climb_points_(i, 0) = msg.points[i].x;
            climb_points_(i, 1) = msg.points[i].y;
        }
        climb_stamp_ns_ = now().nanoseconds();
    }

    // Port of PlanningNode._min_span_map.
    Eigen::ArrayXXf min_span_map() {
        if (!signal_fresh(climb_stamp_ns_, climb_region_ttl_ns_) || climb_points_.rows() == 0) {
            return Eigen::ArrayXXf();
        }
        Eigen::Vector2i shape(grid_shape_[0], grid_shape_[1]);
        Eigen::ArrayXXi seeds = Eigen::ArrayXXi::Zero(shape[0], shape[1]);
        for (int i = 0; i < climb_points_.rows(); ++i) {
            const int r = static_cast<int>(std::floor(
                (climb_points_(i, 0) - origin_[0]) / resolution_));
            const int c = static_cast<int>(std::floor(
                (climb_points_(i, 1) - origin_[1]) / resolution_));
            if (r >= 0 && r < shape[0] && c >= 0 && c < shape[1]) {
                seeds(r, c) = 1;
            }
        }
        if (seeds.count() == 0) {
            return Eigen::ArrayXXf();
        }
        // Grow each seed into a square of the radius.
        const planning::Mask2D region = planning::maximum_filter(
            seeds.cast<bool>(), 2 * climb_region_cells_ + 1);
        Eigen::ArrayXXf out = Eigen::ArrayXXf::Constant(shape[0], shape[1],
                                                        obstacle_config_.min_wall_span_m);
        for (int r = 0; r < shape[0]; ++r) {
            for (int c = 0; c < shape[1]; ++c) {
                if (region(r, c)) out(r, c) = static_cast<float>(climb_min_wall_span_m_);
            }
        }
        return out;
    }

    bool signal_fresh(std::optional<int64_t> stamp_ns, int64_t window_ns) const {
        if (!stamp_ns.has_value()) return false;
        return now().nanoseconds() - *stamp_ns <= window_ns;
    }

    // Port of PlanningNode._open_target_speed.
    double open_target_speed() {
        if (signal_fresh(speed_cap_stamp_ns_, speed_cap_ttl_ns_) &&
            std::isfinite(*speed_cap_)) {
            return std::clamp(*speed_cap_ * capture_speed_gain_, vx_min_, vx_hard_max_);
        }
        return vx_max_;
    }

    void target_pose_callback(const nav_msgs::msg::Odometry& msg) {
        target_pose_ = Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y,
                                       msg.pose.pose.position.z);
    }

    void on_global_route(const nav_msgs::msg::Path& msg) {
        if (msg.poses.size() < 2) {
            has_cached_route_ = false;
            return;
        }
        global_route_map_xy_.resize(static_cast<int>(msg.poses.size()), 2);
        for (size_t i = 0; i < msg.poses.size(); ++i) {
            global_route_map_xy_(static_cast<int>(i), 0) = msg.poses[i].pose.position.x;
            global_route_map_xy_(static_cast<int>(i), 1) = msg.poses[i].pose.position.y;
        }
        has_cached_route_ = true;
    }

    // Port of PlanningNode._route_in_world (TF 'world' <- 'map').
    std::optional<Eigen::MatrixX2d> route_in_world() {
        if (!has_cached_route_ || global_route_map_xy_.rows() == 0) {
            return std::nullopt;
        }
        geometry_msgs::msg::TransformStamped t;
        try {
            t = tf_buffer_->lookupTransform("world", "map", tf2::TimePointZero);
        } catch (const tf2::TransformException&) {
            return std::nullopt;
        }
        const Eigen::Quaterniond quat(t.transform.rotation.w, t.transform.rotation.x,
                                      t.transform.rotation.y, t.transform.rotation.z);
        const Eigen::Matrix3d rot = quat.normalized().toRotationMatrix();
        const Eigen::Vector2d trans(t.transform.translation.x, t.transform.translation.y);
        return (global_route_map_xy_ * rot.topLeftCorner<2, 2>().transpose()).rowwise() + trans.transpose();
    }

    void info_callback(const sensor_msgs::msg::CameraInfo& msg) {
        if (K_.has_value()) return;
        Eigen::Matrix3d k;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                k(r, c) = msg.k[static_cast<size_t>(r * 3 + c)];
            }
        }
        const double fx = k(0, 0);
        const double Tx = msg.p[3];  // right camera's projection matrix
        K_ = k;
        baseline_ = -Tx / fx;
        RCLCPP_INFO(get_logger(), "Camera intrinsics and baseline received. Baseline: %.4fm",
                    baseline_);
        camerainfo_sub_.reset();
    }

    // Port of camera_to_robot_center / _footprint_hits / _front_obstacle_dist —
    // the pure math lives in the planning library.

    // --- per-frame pipeline ------------------------------------------------
    // Port of update_occupancy_grid.
    void update_occupancy_grid(const Eigen::MatrixXf& depth, const Eigen::Matrix4d& T,
                               double fx, double fy, double cx, double cy) {
        const Eigen::Vector3d center =
            origin_ + grid_shape_.cast<double>() * resolution_ / 2.0;
        const Eigen::Vector3d robot_pos = T.topRightCorner<3, 1>();
        const Eigen::Vector3d target_center =
            robot_pos - Eigen::Vector3d(0.0, 0.0, z_grid_drop_);
        const Eigen::Vector3d delta = target_center - center;
        if (delta.norm() > 0.1) {
            const Eigen::Vector3d new_origin =
                target_center - grid_shape_.cast<double>() * resolution_ / 2.0;
            const planning::RolledGrid rolled = planning::roll_occupancy_grid(
                occupancy_grid_, origin_, new_origin, resolution_);
            occupancy_grid_ = rolled.grid;
            origin_ = rolled.origin;
        }
        const planning::OccupancyGrid3D new_occ = planning::run_raycasting_loopy(
            depth, T, grid_shape_, fx, fy, cx, cy, origin_, step_, resolution_);
        occupancy_grid_.scale(0.99);
        occupancy_grid_.add(new_occ);
        occupancy_grid_.clip(-0.2, 0.2);
    }

    // Port of build_obstacle_and_esdf.
    void build_obstacle_and_esdf(const Eigen::Matrix4d& T,
                                 Eigen::ArrayXXf& min_span_out,
                                 planning::Mask2D& obstacle_mask,
                                 Eigen::ArrayXXf& esdf_map) {
        Eigen::ArrayXXf span = min_span_map();
        min_span_out = span;
        const Eigen::ArrayXXf* span_ptr = span.size() > 0 ? &span : nullptr;
        obstacle_mask = planning::build_obstacle_map(
            occupancy_grid_, origin_, resolution_, T(2, 3), obstacle_config_, span_ptr);
        if (obstacle_mask.any()) {
            // The EDT's features are the zero cells of its input, so the ESDF
            // (distance to the nearest obstacle) reads distance_transform_edt(
            // ~obstacle_mask) — pass the inverted mask.
            planning::Mask2D edt_input = planning::Mask2D::Constant(
                obstacle_mask.rows(), obstacle_mask.cols(), true);
            for (int r = 0; r < obstacle_mask.rows(); ++r) {
                for (int c = 0; c < obstacle_mask.cols(); ++c) {
                    if (obstacle_mask(r, c)) edt_input(r, c) = false;
                }
            }
            const planning::EdtResult edt = planning::distance_transform_edt(edt_input);
            esdf_map = (edt.distances * resolution_).cast<float>();
        } else {
            // An empty world would score as solid collision otherwise (the gz
            // sim's first frames).
            esdf_map = Eigen::ArrayXXf::Constant(obstacle_mask.rows(), obstacle_mask.cols(),
                                                 static_cast<float>(free_space_esdf_));
        }
    }

    // Port of generate_trajectories.
    planning::TrajectorySet generate_trajectories(const Eigen::Vector3d& init_p,
                                                  const Eigen::Vector4d& init_q,
                                                  double v_allow) {
        planning::TrajectorySet lib = planning::generate_trajectory_library_3d(
            15, 3.0, traj_dt_, init_p, init_q, v_allow, robot_.max_angular_vel,
            traj_max_len_m_, traj_max_lat_acc_, vx_min_);
        planning::TrajectorySet vocab =
            planning::generate_predefined_trajectory_vocabularies(3.0, traj_dt_, init_p, init_q);
        return planning::concatenate_trajectories(lib, vocab);
    }

    // Port of sync_callback.
    void sync_callback(Image::ConstSharedPtr depth_msg,
                       nav_msgs::msg::Odometry::ConstSharedPtr odom_msg) {
        if (!K_.has_value()) {
            if (no_k_due()) {
                RCLCPP_WARN(get_logger(),
                            "[planning] K is None: no /camera/camera/infra2/camera_info yet, planner idle");
            }
            return;
        }
        const Eigen::MatrixXf depth = depth_to_eigen(*depth_msg);
        const Eigen::Matrix4d T = odom_to_T(*odom_msg);
        const double fx = (*K_)(0, 0), fy = (*K_)(1, 1);
        const double cx = (*K_)(0, 2), cy = (*K_)(1, 2);

        update_occupancy_grid(depth, T, fx, fy, cx, cy);

        if (occupancy_cloud_pub_->get_subscription_count() > 0) {
            publish_3d_occupancy_cloud();
        }

        Eigen::ArrayXXf min_span;
        planning::Mask2D obstacle_mask;
        Eigen::ArrayXXf esdf_map;
        build_obstacle_and_esdf(T, min_span, obstacle_mask, esdf_map);

        publish_3d_occupancy_cloud_with_esdf(esdf_map);
        publish_height_map(esdf_map, depth_msg->header);
        publish_2d_occupancy_grid(esdf_map, depth_msg->header.stamp,
                                  grid_shape_[2] * resolution_ / 2.0);
        publish_obstacle_mask(obstacle_mask, depth_msg->header.stamp);
        publish_footprint(T, depth_msg->header.stamp);

        const Eigen::Vector3d init_p = planning::camera_to_robot_center(T, robot_);
        const Eigen::Vector4d init_q(odom_msg->pose.pose.orientation.x,
                                     odom_msg->pose.pose.orientation.y,
                                     odom_msg->pose.pose.orientation.z,
                                     odom_msg->pose.pose.orientation.w);
        // Forward clearance drives both the peak-speed schedule and the reverse gate.
        const double front_clearance =
            planning::front_obstacle_dist(T, obstacle_mask, origin_, resolution_, robot_, clear_scan_m_);
        const double v_open = open_target_speed();
        const double v_allow = planning::speed_from_clearance(
            front_clearance, std::abs(last_param_[0]), v_open, vx_min_, clear_c0_m_,
            clear_open_m_, t_react_s_);

        if (status_due()) {
            const std::string goal = !target_pose_.has_value()
                                         ? std::string("none")
                                         : fmt_goal(*target_pose_);
            RCLCPP_INFO(get_logger(),
                        "[planning] goal=%s pose=(%.2f,%.2f) v_allow=%.2f front_clr=%.2f "
                        "obs=%d route=%c sel=(%.2f,%.2f)",
                        goal.c_str(), init_p[0], init_p[1], v_allow, front_clearance,
                        static_cast<int>(obstacle_mask.count()),
                        has_cached_route_ ? 'y' : 'n', last_param_[0], last_param_[1]);
        }

        // The library exists only to pick a trajectory toward a goal.
        if (!target_pose_.has_value()) {
            return;
        }
        const planning::TrajectorySet trajectories =
            generate_trajectories(init_p, init_q, v_allow);

        std::optional<Eigen::MatrixX2d> route_xy = route_in_world();
        const planning::RouteFields fields = planning::build_route_fields(
            route_xy.has_value() ? *route_xy : Eigen::MatrixX2d(),
            Eigen::Vector2i(esdf_map.rows(), esdf_map.cols()), origin_, resolution_);

        const planning::EsdfScoreResult scored = planning::score_trajectories(
            trajectories, trajectories.params, esdf_map, fields.path_dist_map,
            fields.remaining_map, fields.route_heading_map, origin_, resolution_, robot_);

        int n_fwd_ok = 0;
        for (int i = 0; i < trajectories.num_trajectories; ++i) {
            if (trajectories.params(i, 0) > 1e-3 &&
                scored.scores[static_cast<size_t>(i)] != std::numeric_limits<double>::infinity()) {
                ++n_fwd_ok;
            }
        }
        const bool should_reverse =
            planning::reverse_armed(front_clearance, n_fwd_ok, resolution_);

        const bool all_inf = std::all_of(
            scored.scores.begin(), scored.scores.end(),
            [](double s) { return s == std::numeric_limits<double>::infinity(); });
        if (all_inf) {
            const Eigen::Vector3d center = planning::camera_to_robot_center(T, robot_);
            const int cxi = static_cast<int>((center[0] - origin_[0]) / resolution_);
            const int cyi = static_cast<int>((center[1] - origin_[1]) / resolution_);
            const double esdf_center =
                (0 <= cxi && cxi < esdf_map.rows() && 0 <= cyi && cyi < esdf_map.cols())
                    ? esdf_map(cxi, cyi)
                    : -1.0;
            const planning::FootprintHits hits =
                planning::footprint_hits(T, obstacle_mask, origin_, resolution_, robot_);
            RCLCPP_WARN(
                get_logger(),
                "All trajectories in collision. obst_cells=%d front_clearance=%.2f "
                "ESDF@center=%.2f should_reverse=%d %s",
                static_cast<int>(obstacle_mask.count()), front_clearance, esdf_center,
                should_reverse ? 1 : 0, hits_report(hits).c_str());
            // Nothing is published, and that is the honest answer here.
            return;
        }

        const int top_index = planning::select_trajectory(
            trajectories, trajectories.params, scored, fields.has_route, target_pose_,
            last_param_, should_reverse, weights_);
        last_param_ = Eigen::Vector2d(trajectories.params(top_index, 0),
                                      trajectories.params(top_index, 1));

        // A goal-holding standstill is legal output, but when it persists it reads
        // as a silent freeze — name the losing arcs (stuck-by-cost log).
        if (std::abs(last_param_[0]) < 1e-3 && std::abs(last_param_[1]) < 1e-3) {
            ++stuck_cycles_;
            if (stuck_cycles_ == 1 || stuck_cycles_ % 25 == 0) {
                double best_cost = std::numeric_limits<double>::infinity();
                int best_i = -1;
                for (int i = 0; i < trajectories.num_trajectories; ++i) {
                    if (trajectories.params(i, 0) > 1e-3 &&
                        scored.scores[static_cast<size_t>(i)] !=
                            std::numeric_limits<double>::infinity()) {
                        const double cost = planning::trajectory_cost(
                            i, trajectories, trajectories.params, scored, fields.has_route,
                            target_pose_, last_param_, should_reverse, weights_);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_i = i;
                        }
                    }
                }
                const std::string detail =
                    best_i >= 0
                        ? fmt_stuck(best_i, best_cost, trajectories)
                        : std::string("no collision-free moving arc");
                RCLCPP_WARN(get_logger(),
                            "stuck by cost (%d cycles): goal=(%.2f,%.2f) fwd_ok=%d "
                            "front_clr=%.2f %s",
                            stuck_cycles_, target_pose_->x(), target_pose_->y(), n_fwd_ok,
                            front_clearance, detail.c_str());
            }
        } else {
            stuck_cycles_ = 0;
        }

        RCLCPP_INFO(
            get_logger(),
            "sel vx=%.2f omega=%.2f fwd_ok=%d goal_err=%.0fdeg route_err=%.0fdeg "
            "v_allow=%.2f front_clr=%.2f should_reverse=%d climb_cells=%d",
            last_param_[0], last_param_[1], n_fwd_ok,
            rad2deg(planning::end_heading_error(trajectories.pose(top_index, 0), *target_pose_)),
            rad2deg(scored.end_heading_errs[static_cast<size_t>(top_index)]),
            v_allow, front_clearance, should_reverse ? 1 : 0,
            min_span.size() > 0
                ? static_cast<int>((min_span > obstacle_config_.min_wall_span_m).count())
                : 0);

        // velocity feedforward: yaw rate straight from the trajectory's own world
        // poses (d(heading)/dt over the first step), not the lattice omega param.
        const double dh = planning::heading_of_pose7(trajectories.pose(top_index, 1)) -
                          planning::heading_of_pose7(trajectories.pose(top_index, 0));
        (void)std::atan2(std::sin(dh), std::cos(dh));  // same fold the Python takes
        publish_selected_path(trajectories, top_index, depth_msg->header);
    }

    static double rad2deg(double rad) { return rad * 180.0 / M_PI; }

    static std::string fmt_goal(const Eigen::Vector3d& goal) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "(%.2f,%.2f)", goal.x(), goal.y());
        return buf;
    }

    std::string fmt_stuck(int best_i, double best_cost, const planning::TrajectorySet& trajs) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "best moving vx=%.2f omega=%.2f cost=%.1f",
                      trajs.params(best_i, 0), trajs.params(best_i, 1), best_cost);
        return buf;
    }

    static std::string hits_report(const planning::FootprintHits& hits) {
        if (hits.hits.empty()) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "body_hits=0/%d", hits.n_samples);
            return buf;
        }
        std::string out = "body_hits=" + std::to_string(hits.hits.size()) + "/" +
                          std::to_string(hits.n_samples) + " at";
        const size_t shown = std::min<size_t>(hits.hits.size(), 8);
        char cell[32];
        for (size_t i = 0; i < shown; ++i) {
            std::snprintf(cell, sizeof(cell), " (%+.2f,%+.2f)", hits.hits[i].first,
                          hits.hits[i].second);
            out += cell;
        }
        if (hits.hits.size() > 8) out += " ...";
        return out;
    }

    // logsetup::every(1.0) equivalent.
    bool status_due() {
        const rclcpp::Time t = now();
        if (last_status_.nanoseconds() == 0 ||
            (t - last_status_).seconds() >= 1.0) {
            last_status_ = t;
            return true;
        }
        return false;
    }
    bool no_k_due() {
        const rclcpp::Time t = now();
        if (last_no_k_.nanoseconds() == 0 || (t - last_no_k_).seconds() >= 1.0) {
            last_no_k_ = t;
            return true;
        }
        return false;
    }

    // --- publishers --------------------------------------------------------
    // Publish profiling (diagnostic): times each publish() (serialization +
    // DDS/intra-proc write) and dumps n/avg/max per topic every 5 s of
    // steady-clock. All publishes run on the executor thread — no lock needed.
    struct PublishStat {
        int count = 0;
        double total_ms = 0.0;
        double max_ms = 0.0;
    };

    void profile_publish(const char* topic, const std::function<void()>& publish_fn) {
        const auto t0 = std::chrono::steady_clock::now();
        publish_fn();
        const auto t1 = std::chrono::steady_clock::now();
        PublishStat& stat = publish_stats_[topic];
        ++stat.count;
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stat.total_ms += ms;
        stat.max_ms = std::max(stat.max_ms, ms);
        if (std::chrono::duration<double>(t1 - profile_window_start_).count() >= 5.0) {
            for (const auto& [name, st] : publish_stats_) {
                RCLCPP_INFO(get_logger(),
                            "[publish-profile] %-28s n=%-5d avg=%7.2fms max=%8.2fms",
                            name.c_str(), st.count,
                            st.count > 0 ? st.total_ms / st.count : 0.0, st.max_ms);
            }
            publish_stats_.clear();
            profile_window_start_ = t1;
        }
    }

    // Decimation is load-bearing: cmd_vel_control reads dt off this Path
    // (PATH_POSE_STRIDE = 10).
    void publish_selected_path(const planning::TrajectorySet& trajectories, int index,
                               const std_msgs::msg::Header& header) {
        nav_msgs::msg::Path path;
        path.header = header;
        path.header.frame_id = "world";
        for (int j = 0; j < trajectories.num_steps; j += 10) {
            const auto pose = trajectories.pose(index, j);
            geometry_msgs::msg::PoseStamped stamped;
            stamped.header = header;
            stamped.pose.position.x = pose[0];
            stamped.pose.position.y = pose[1];
            stamped.pose.position.z = pose[2];
            stamped.pose.orientation.x = pose[3];
            stamped.pose.orientation.y = pose[4];
            stamped.pose.orientation.z = pose[5];
            stamped.pose.orientation.w = pose[6];
            path.poses.push_back(stamped);
        }
        profile_publish("trajectory_path", [&] {
            path_pub_->publish(std::make_unique<nav_msgs::msg::Path>(std::move(path)));
        });
    }

    void publish_footprint(const Eigen::Matrix4d& T, const builtin_interfaces::msg::Time& stamp) {
        const Eigen::Vector3d forward =
            T.topLeftCorner<3, 3>() * Eigen::Vector3d(0.0, 0.0, 1.0);
        const Eigen::Vector3d left =
            T.topLeftCorner<3, 3>() * Eigen::Vector3d(1.0, 0.0, 0.0);
        const Eigen::Vector3d center = planning::camera_to_robot_center(T, robot_);
        const auto [fl, rl, hw] = robot_.footprint_from_control();
        const Eigen::Vector3d corners[4] = {
            center + forward * fl + left * hw,
            center + forward * fl - left * hw,
            center - forward * rl - left * hw,
            center - forward * rl + left * hw,
        };
        sensor_msgs::msg::PointCloud cloud;
        cloud.header.stamp = stamp;
        cloud.header.frame_id = "world";
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 21; ++k) {
                const double t = k / 20.0;
                const Eigen::Vector3d p = (1.0 - t) * corners[i] + t * corners[(i + 1) % 4];
                geometry_msgs::msg::Point32 pt;
                pt.x = static_cast<float>(p[0]);
                pt.y = static_cast<float>(p[1]);
                pt.z = static_cast<float>(p[2]);
                cloud.points.push_back(pt);
            }
        }
        profile_publish("footprint", [&] {
            footprint_pub_->publish(std::make_unique<sensor_msgs::msg::PointCloud>(std::move(cloud)));
        });
    }

    void publish_obstacle_mask(const planning::Mask2D& mask, const builtin_interfaces::msg::Time& stamp) {
        nav_msgs::msg::OccupancyGrid grid;
        grid.header.stamp = stamp;
        grid.header.frame_id = "world";
        grid.info.resolution = resolution_;
        grid.info.width = static_cast<uint32_t>(mask.cols());
        grid.info.height = static_cast<uint32_t>(mask.rows());
        grid.info.origin.position.x = origin_[0];
        grid.info.origin.position.y = origin_[1];
        grid.info.origin.position.z =
            origin_[2] + grid_shape_[2] * resolution_ / 2.0;
        grid.info.origin.orientation.w = 1.0;
        // np.where(mask, 100, 0).ravel(order="F")
        grid.data.resize(static_cast<size_t>(mask.rows()) * mask.cols());
        size_t idx = 0;
        for (int c = 0; c < mask.cols(); ++c) {
            for (int r = 0; r < mask.rows(); ++r) {
                grid.data[idx++] = mask(r, c) ? 100 : 0;
            }
        }
        profile_publish("obstacle_mask", [&] {
            obstacle_mask_pub_->publish(
                std::make_unique<nav_msgs::msg::OccupancyGrid>(std::move(grid)));
        });
    }

    void publish_height_map(const Eigen::ArrayXXf& esdf_map, const std_msgs::msg::Header& header) {
        cv::Mat normalized(static_cast<int>(esdf_map.rows()), static_cast<int>(esdf_map.cols()),
                           CV_8UC1);
        for (int r = 0; r < esdf_map.rows(); ++r) {
            for (int c = 0; c < esdf_map.cols(); ++c) {
                const double v = std::clamp(esdf_map(r, c) / 2.0 * 255.0, 0.0, 255.0);
                normalized.at<uint8_t>(r, c) = static_cast<uint8_t>(v);
            }
        }
        cv::Mat color;
        cv::applyColorMap(normalized, color, cv::COLORMAP_JET);
        auto msg = cv_bridge::CvImage(header, "bgr8", color).toImageMsg();
        profile_publish("height_map", [&] { height_map_pub_->publish(*msg); });
    }

    void publish_2d_occupancy_grid(const Eigen::ArrayXXf& esdf_map,
                                   const builtin_interfaces::msg::Time& stamp, double z_offset) {
        nav_msgs::msg::OccupancyGrid grid;
        grid.header.stamp = stamp;
        grid.header.frame_id = "world";
        grid.info.resolution = resolution_;
        grid.info.width = static_cast<uint32_t>(esdf_map.cols());
        grid.info.height = static_cast<uint32_t>(esdf_map.rows());
        grid.info.origin.position.x = origin_[0];
        grid.info.origin.position.y = origin_[1];
        grid.info.origin.position.z = origin_[2] + z_offset;
        grid.info.origin.orientation.w = 1.0;
        grid.data.resize(static_cast<size_t>(esdf_map.rows()) * esdf_map.cols());
        size_t idx = 0;
        for (int c = 0; c < esdf_map.cols(); ++c) {
            for (int r = 0; r < esdf_map.rows(); ++r) {
                int8_t value;
                const float d = esdf_map(r, c);
                if (d <= 0.0f) {
                    value = 100;
                } else {
                    const int scaled =
                        static_cast<int>((1.0 - d / 0.5) * 120.0);
                    value = static_cast<int8_t>(std::clamp(scaled, 0, 120));
                }
                grid.data[idx++] = value;
            }
        }
        profile_publish("occupancy_grid", [&] {
            occupancy_grid_pub_->publish(
                std::make_unique<nav_msgs::msg::OccupancyGrid>(std::move(grid)));
        });
    }

    void publish_3d_occupancy_cloud() {
        // Port of publish_3d_occupancy_cloud (occupied = grid > 0.1, xyz32 cloud).
        std::vector<Eigen::Vector3f> points;
        for (int x = 0; x < occupancy_grid_.nx(); ++x) {
            for (int y = 0; y < occupancy_grid_.ny(); ++y) {
                for (int z = 0; z < occupancy_grid_.nz(); ++z) {
                    if (occupancy_grid_(x, y, z) > 0.1) {
                        points.emplace_back(
                            static_cast<float>(origin_[0] + x * resolution_),
                            static_cast<float>(origin_[1] + y * resolution_),
                            static_cast<float>(origin_[2] + z * resolution_));
                    }
                }
            }
        }
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = now();
        cloud.header.frame_id = "world";
        cloud.height = 1;
        cloud.width = static_cast<uint32_t>(points.size());
        cloud.is_dense = true;
        cloud.is_bigendian = false;
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(static_cast<size_t>(points.size()));
        sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x"), iy(cloud, "y"), iz(cloud, "z");
        for (const auto& p : points) {
            *ix = p.x(); ++ix;
            *iy = p.y(); ++iy;
            *iz = p.z(); ++iz;
        }
        profile_publish("occupied_voxels", [&] {
            occupancy_cloud_pub_->publish(
                std::make_unique<sensor_msgs::msg::PointCloud2>(std::move(cloud)));
        });
    }

    void publish_3d_occupancy_cloud_with_esdf(const Eigen::ArrayXXf& esdf_map) {
        // Port of publish_3d_occupancy_cloud_with_esdf: the ground plane z = +2
        // voxels, coloured by ESDF through the JET colormap. One colormap call
        // over all values (the Python applies it to the whole vector too).
        const int X = grid_shape_[0], Y = grid_shape_[1];
        const double max_dist = 1.0;
        const int total = X * Y;
        cv::Mat values(1, total, CV_8UC1);
        {
            uint8_t* p = values.data;
            for (int x = 0; x < X; ++x) {
                for (int y = 0; y < Y; ++y) {
                    const float dist = std::clamp(esdf_map(x, y), 0.0f, static_cast<float>(max_dist));
                    *p++ = static_cast<uint8_t>((1.0 - dist / max_dist) * 255.0);
                }
            }
        }
        cv::Mat colors;
        cv::applyColorMap(values, colors, cv::COLORMAP_JET);  // (1, total) CV_8UC3 (b,g,r)
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = now();
        cloud.header.frame_id = "world";
        cloud.height = 1;
        cloud.width = static_cast<uint32_t>(total);
        cloud.is_dense = true;
        cloud.is_bigendian = false;
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2Fields(
            4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
            sensor_msgs::msg::PointField::FLOAT32, "z", 1,
            sensor_msgs::msg::PointField::FLOAT32, "rgb", 1,
            sensor_msgs::msg::PointField::UINT32);
        modifier.resize(static_cast<size_t>(total));
        sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x"), iy(cloud, "y"), iz(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint32_t> irgb(cloud, "rgb");
        int i = 0;
        for (int x = 0; x < X; ++x) {
            for (int y = 0; y < Y; ++y, ++i) {
                const cv::Vec3b& bgr = colors.at<cv::Vec3b>(0, i);
                const uint32_t rgb = (static_cast<uint32_t>(bgr[2]) << 16) |
                                     (static_cast<uint32_t>(bgr[1]) << 8) | bgr[0];
                *ix = static_cast<float>(origin_[0] + x * resolution_); ++ix;
                *iy = static_cast<float>(origin_[1] + y * resolution_); ++iy;
                *iz = static_cast<float>(origin_[2] + 2 * resolution_); ++iz;
                *irgb = rgb; ++irgb;
            }
        }
        profile_publish("occupied_voxels_with_esdf", [&] {
            occupancy_cloud_esdf_pub_->publish(
                std::make_unique<sensor_msgs::msg::PointCloud2>(std::move(cloud)));
        });
    }

    // --- small adapters ----------------------------------------------------
    static Eigen::MatrixXf depth_to_eigen(const Image& msg) {
        const cv::Mat img = cv_bridge::toCvCopy(msg, "32FC1")->image;
        Eigen::MatrixXf out(img.rows, img.cols);
        for (int r = 0; r < img.rows; ++r) {
            for (int c = 0; c < img.cols; ++c) {
                out(r, c) = img.at<float>(r, c);
            }
        }
        return out;
    }

    // msg2np: position + normalized quaternion -> 4x4.
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

    // --- state -------------------------------------------------------------
    const core::RobotConfig& robot_ = core::robot_config_from_env();
    core::ObstacleConfig obstacle_config_;
    double resolution_ = 0.05;
    Eigen::Vector3i grid_shape_{0, 0, 0};
    double z_grid_drop_ = 0.0;
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    double free_space_esdf_ = 0.0;
    int step_ = 4;
    double traj_dt_ = 0.1;  // matches the library / vocab dt

    planning::OccupancyGrid3D occupancy_grid_;
    std::optional<Eigen::Matrix3d> K_;
    double baseline_ = 0.0;
    Eigen::Vector2d last_param_ = Eigen::Vector2d::Zero();  // (vx, omega)
    int stuck_cycles_ = 0;
    planning::DwaWeights weights_;

    std::optional<Eigen::Vector3d> target_pose_;
    Eigen::MatrixX2d global_route_map_xy_;
    bool has_cached_route_ = false;

    Eigen::MatrixXd climb_points_{0, 2};
    std::optional<int64_t> climb_stamp_ns_;
    std::optional<float> speed_cap_;
    std::optional<int64_t> speed_cap_stamp_ns_;

    // cached parameters
    double vx_max_ = 0.6, vx_hard_max_ = 1.0, vx_min_ = 0.2;
    double clear_c0_m_ = 0.35, clear_open_m_ = 1.0, clear_scan_m_ = 2.0;
    double t_react_s_ = 0.2, traj_max_len_m_ = 2.5, traj_max_lat_acc_ = 0.5;
    int climb_region_cells_ = 15;
    int64_t climb_region_ttl_ns_ = 3'000'000'000;
    double climb_min_wall_span_m_ = 0.2;
    double capture_speed_gain_ = 1.0;
    int64_t speed_cap_ttl_ns_ = 2'000'000'000;

    rclcpp::Time last_status_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_no_k_{0, 0, RCL_ROS_TIME};

    using Sync = Synchronizer<ExactTime<Image, nav_msgs::msg::Odometry>>;
    message_filters::Subscriber<Image> depth_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> pose_sub_;
    std::shared_ptr<Sync> sync_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<Image>::SharedPtr height_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr obstacle_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr footprint_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occupancy_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occupancy_cloud_esdf_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;

    // publish profiling state (see profile_publish)
    std::unordered_map<std::string, PublishStat> publish_stats_;
    std::chrono::steady_clock::time_point profile_window_start_ =
        std::chrono::steady_clock::now();
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camerainfo_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr target_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr poi_change_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_route_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr climb_region_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_cap_sub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::Node::SharedPtr tf_node_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

// main.cpp's factory contract.
std::shared_ptr<rclcpp::Node> make_planning(const rclcpp::NodeOptions& options) {
    return std::make_shared<PlanningComponent>(options);
}

}  // namespace tinynav
