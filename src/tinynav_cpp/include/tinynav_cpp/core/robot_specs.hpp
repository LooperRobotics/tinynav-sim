// Port of reference/tinynav/core/robot_specs.py — header-only.
// Dataclasses become plain structs with the Python defaults as member
// initializers; the named configs are inline accessors over function-local
// statics (no constexpr: std::string members), and the module-level
// ROBOT_TYPE lookup becomes robot_config().
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinynav::core {

// Port of reference/tinynav/core/robot_specs.py::ObstacleConfig
// Height band + occupancy filters used by planning_node.build_obstacle_map.
// The z-band is relative to camera height; see the Python docstring for why
// these numbers are per-robot.
struct ObstacleConfig {
    double robot_z_bottom = -0.45;
    double robot_z_top = 0.2;
    double occ_threshold = 0.05;
    double min_wall_span_m = 0.05;
    // Only cells whose lowest occupied voxel sits within this of
    // robot_z_bottom are span-filtered; cells starting above the band are
    // floating obstacles and keep a single-voxel noise floor.
    double ground_band_m = 0.3;
    int dilation_cells = 0;
};

// Python footprint_from_control() -> (front_len, rear_len, half_w) tuple.
struct FootprintFromControl {
    double front_len;
    double rear_len;
    double half_w;
};

// Port of reference/tinynav/core/robot_specs.py::RobotConfig
// Robot geometry + velocity limits. Body frame: +x forward, +y left.
struct RobotConfig {
    std::string name = "go2";
    std::string shape = "square";  // "square" | "circle"
    double length = 0.7;
    double width = 0.3;
    double radius = 0.3;
    double camera_x = 0.35;
    double camera_y = 0.0;
    double control_x = 0.0;
    double control_y = 0.0;
    double safety_radius = 0.1;
    double min_linear_vel = 0.1;
    double max_linear_vel = 1.0;
    double min_angular_vel = 0.1;
    double max_angular_vel = 0.75;
    ObstacleConfig obstacle;

    // Offset [left, up, forward] from control center to camera in body frame.
    Eigen::Vector3d cam_offset_3d() const {
        return {camera_y - control_y, 0.0, camera_x - control_x};
    }

    // (half_length, half_width); a circle reads as (radius, radius).
    std::pair<double, double> half_size() const {
        if (shape == "circle") {
            return {radius, radius};
        }
        return {length / 2.0, width / 2.0};
    }

    // Footprint relative to the control center.
    FootprintFromControl footprint_from_control() const {
        const auto [hl, hw] = half_size();
        return {hl - control_x, hl + control_x, hw};
    }
};

// Port of reference/tinynav/core/robot_specs.py::GO2_CONFIG
inline const RobotConfig& go2_config() {
    static const RobotConfig config = [] {
        RobotConfig c;
        c.name = "go2";
        c.shape = "square";
        c.length = 0.6;
        c.width = 0.3;
        c.camera_x = 0.35;
        c.camera_y = 0.0;
        c.control_x = 0.05;
        c.control_y = 0.0;
        c.safety_radius = 0.1;
        return c;
    }();
    return config;
}

// Port of reference/tinynav/core/robot_specs.py::GO2W_CONFIG
inline const RobotConfig& go2w_config() {
    static const RobotConfig config = [] {
        RobotConfig c;
        c.name = "go2w";
        c.shape = "square";
        c.length = 0.6;
        c.width = 0.3;
        c.camera_x = 0.35;
        c.camera_y = 0.0;
        c.control_x = 0.05;
        c.control_y = 0.0;
        c.safety_radius = 0.1;
        return c;
    }();
    return config;
}

// Port of reference/tinynav/core/robot_specs.py::B2_CONFIG
inline const RobotConfig& b2_config() {
    static const RobotConfig config = [] {
        RobotConfig c;
        c.name = "b2";
        c.shape = "square";
        c.length = 0.8;
        c.width = 0.3;
        c.camera_x = 0.5;
        c.camera_y = 0.0;
        c.control_x = 0.2;
        c.control_y = 0.0;
        c.safety_radius = 0.1;
        // Taller than the default band covers: the operator's call, 2026-09-03.
        c.obstacle.robot_z_bottom = -0.6;
        c.obstacle.robot_z_top = 0.6;
        return c;
    }();
    return config;
}

// Port of reference/tinynav/core/robot_specs.py::B2W_CONFIG
inline const RobotConfig& b2w_config() {
    static const RobotConfig config = [] {
        RobotConfig c;
        c.name = "b2w";
        c.shape = "square";
        c.length = 0.8;
        c.width = 0.3;
        c.camera_x = 0.5;
        c.camera_y = 0.0;
        c.control_x = 0.0;
        c.control_y = 0.0;
        c.safety_radius = 0.1;
        return c;
    }();
    return config;
}

// Port of reference/tinynav/core/robot_specs.py::G1_CONFIG
inline const RobotConfig& g1_config() {
    static const RobotConfig config = [] {
        RobotConfig c;
        c.name = "g1";
        c.shape = "square";
        c.length = 0.3;
        c.width = 0.5;
        c.camera_x = 0.1;
        c.camera_y = 0.0;
        c.control_x = 0.0;
        c.control_y = 0.0;
        c.safety_radius = 0.15;
        c.min_linear_vel = 0.2;
        c.min_angular_vel = 0.3;
        return c;
    }();
    return config;
}

// Port of reference/tinynav/core/robot_specs.py::LEKIWI_CONFIG
inline const RobotConfig& lekiwi_config() {
    static const RobotConfig config = [] {
        RobotConfig c;
        c.name = "lekiwi";
        c.shape = "circle";
        c.length = 0.2;
        c.width = 0.2;
        c.radius = 0.1;
        c.camera_x = 0.09;
        c.camera_y = 0.0;
        c.control_x = 0.0;
        c.control_y = 0.0;
        c.safety_radius = 0.05;
        c.min_linear_vel = 0.1;
        c.max_linear_vel = 0.5;
        c.min_angular_vel = 0.2;
        c.max_angular_vel = 1.0;
        c.obstacle.robot_z_bottom = -0.15;
        c.obstacle.robot_z_top = 0.15;
        return c;
    }();
    return config;
}

// Port of reference/tinynav/core/robot_specs.py's module-level lookup
// (`globals()[f"{ROBOT_TYPE.upper()}_CONFIG"]`): case- and whitespace-
// insensitive by-name query; unknown names throw std::invalid_argument,
// matching Python's ValueError.
inline const RobotConfig& robot_config(const std::string& robot_type) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    const auto first = std::find_if(robot_type.begin(), robot_type.end(), not_space);
    std::string key;
    if (first != robot_type.end()) {
        const auto last = std::find_if(robot_type.rbegin(), robot_type.rend(), not_space).base();
        key.assign(first, last);
    }
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    if (key == "go2") return go2_config();
    if (key == "go2w") return go2w_config();
    if (key == "b2") return b2_config();
    if (key == "b2w") return b2w_config();
    if (key == "g1") return g1_config();
    if (key == "lekiwi") return lekiwi_config();
    throw std::invalid_argument("Unsupported ROBOT_TYPE: '" + robot_type + "'");
}

// The module-level ROBOT_CONFIG: ROBOT_TYPE env var, default "go2".
inline const RobotConfig& robot_config_from_env() {
    const char* env = std::getenv("ROBOT_TYPE");
    return robot_config(env != nullptr ? env : "go2");
}

}  // namespace tinynav::core
