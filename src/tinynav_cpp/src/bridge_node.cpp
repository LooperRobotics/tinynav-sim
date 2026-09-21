// Port of tinynav-pilot/tool/looper_bridge_node.py for the split-site
// deployment — RELAY MODE. The original bridge consumes the camera box's
// per-frame products (/camera/camera/{depth,vio_image,infra1 image} synced,
// vio_100hz) and does the keyframing itself (0.03 m / 1 deg / 3 s criteria,
// depth 16U->32F conversion, vio_100hz -> /slam/odometry): there, the
// camera runs VIO and the bridge produces the final keyframe products. In
// the sim split the "camera" is our own perception component, which already
// emits the final products — so this bridge runs as a QoS-aware relay over
// the SAME publication contract: every entry re-publishes one crossing
// topic under its canonical local name, stamp untouched.
//
// Why a relay at all (the fleet's bandwidth argument): DDS unicasts one copy
// per subscriber, so a crossing topic with N local consumers costs N× link
// bandwidth. The bridge is the ONLY subscriber of its crossing topics; the
// canonical names exist only on this side (the x86 site remaps perception's
// /slam/* products into the camera-box namespace /camera/camera/slam/*), so
// nothing double-matches. imu, infra2 camera_info and /clock cross directly
// on purpose: small, single Orin-side consumer each, and no local republish
// — relaying them would only duplicate them (the rename-exists-to-prevent-
// collision rule, see docs).
//
// Separate process on purpose: it is the link endpoint — a stack crash must
// not take the link down with it (fleet lesson: anything owned by the
// backend dies with the crash it was supposed to record).
//
// Relay QoS: keyframe topics RELIABLE (mapping's ExactTime sync must not
// drop one of the trio); /slam/depth best-effort both sides (planning
// consumes latest-only, drops are its semantics).
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "tinynav_cpp/logging_setup.hpp"

namespace {

struct RelaySpec {
    std::string type;
    std::string from;
    std::string to;
    int depth = 50;
    bool reliable = true;
};

std::vector<RelaySpec> load_relay_table(const std::string& path) {
    YAML::Node config = YAML::LoadFile(path);
    std::vector<RelaySpec> table;
    for (const auto& entry : config["relay"]) {
        RelaySpec spec;
        spec.type = entry["type"].as<std::string>();
        spec.from = entry["from"].as<std::string>();
        spec.to = entry["to"].as<std::string>();
        if (entry["depth"]) spec.depth = entry["depth"].as<int>();
        if (entry["reliable"]) spec.reliable = entry["reliable"].as<bool>();
        table.push_back(spec);
    }
    return table;
}

class LooperBridgeNode : public rclcpp::Node {
  public:
    explicit LooperBridgeNode(const std::vector<RelaySpec>& table)
        : Node("looper_bridge_node") {
        for (const auto& spec : table) {
            rclcpp::QoS qos(rclcpp::KeepLast(spec.depth));
            if (!spec.reliable) qos.best_effort();
            if (spec.type == "Image") {
                image_pubs_.push_back(create_publisher<sensor_msgs::msg::Image>(
                    spec.to, qos));
                image_subs_.push_back(create_subscription<sensor_msgs::msg::Image>(
                    spec.from, qos,
                    [pub = image_pubs_.back()](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                        pub->publish(*msg);
                    }));
            } else if (spec.type == "Odometry") {
                odom_pubs_.push_back(create_publisher<nav_msgs::msg::Odometry>(
                    spec.to, qos));
                odom_subs_.push_back(create_subscription<nav_msgs::msg::Odometry>(
                    spec.from, qos,
                    [pub = odom_pubs_.back()](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                        pub->publish(*msg);
                    }));
            } else {
                RCLCPP_FATAL(get_logger(), "relay entry %s: unsupported type %s",
                             spec.to.c_str(), spec.type.c_str());
                throw std::runtime_error("unsupported relay type " + spec.type);
            }
            RCLCPP_INFO(get_logger(), "relay %s -> %s (%s, %s)", spec.from.c_str(),
                        spec.to.c_str(), spec.type.c_str(),
                        spec.reliable ? "reliable" : "best_effort");
        }
        RCLCPP_INFO(get_logger(),
                    "%zu relay entries up; imu / infra2 camera_info / clock cross directly",
                    table.size());
    }

  private:
    std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_pubs_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
    std::vector<rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr> odom_pubs_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> odom_subs_;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    tinynav::logging_setup::install();  // tag: looper_bridge (fleet-identical)

    const char* config = argc > 1 ? argv[1] : std::getenv("TINYNAV_BRIDGE_CONFIG");
    if (config == nullptr || *config == '\0') {
        std::fprintf(stderr,
                     "usage: tinynav_bridge <relay.yaml>  (or TINYNAV_BRIDGE_CONFIG)\n");
        return 1;
    }
    try {
        const auto table = load_relay_table(config);
        rclcpp::spin(std::make_shared<LooperBridgeNode>(table));
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("looper_bridge"), "bridge failed: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
