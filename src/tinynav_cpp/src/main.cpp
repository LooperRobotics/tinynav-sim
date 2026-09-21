// Single-process host for all tinynav components (intra-process comms =
// zero-copy pointer passing between them; DDS only at the process boundary).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>

#include "tinynav_cpp/logging_setup.hpp"

// Component factories are linked from the component shared libraries; each
// returns its node as the rclcpp::Node base (the concrete classes stay
// incomplete here, so the shared_ptr conversion happens in the .so).
// The factories live in namespace tinynav alongside their components.
namespace tinynav {
std::shared_ptr<rclcpp::Node> make_imu_propagator(const rclcpp::NodeOptions&);
std::shared_ptr<rclcpp::Node> make_perception(const rclcpp::NodeOptions&);
std::shared_ptr<rclcpp::Node> make_mapping(const rclcpp::NodeOptions&);
std::shared_ptr<rclcpp::Node> make_planning(const rclcpp::NodeOptions&);
}  // namespace tinynav
using tinynav::make_imu_propagator;
using tinynav::make_perception;
using tinynav::make_mapping;
using tinynav::make_planning;

// Component subset selection for the split-site deployment: the x86 site
// runs perception next to the sim, the Orin site runs imu+mapping+planning
// behind the looper_bridge relay. TINYNAV_COMPONENTS is a comma list; the
// default is the full single-process stack. Unknown names fail loudly — a
// typo'd site config must not silently drop a component.
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  // Port of reference/tinynav/core/logsetup.py: per-tag day files, size
  // fuse, 14-day retention, console tee — routed by logger name through one
  // rcutils hook, so the RCLCPP_* call sites below stay untouched.
  tinynav::logging_setup::install();

  const char* kAll[] = {"imu", "perception", "mapping", "planning"};
  std::string wanted = "imu,perception,mapping,planning";
  if (const char* env = std::getenv("TINYNAV_COMPONENTS"); env != nullptr && *env != '\0') {
    wanted = env;
  }
  std::set<std::string> selected;
  std::vector<std::string> unknown;
  for (size_t pos = 0; pos < wanted.size();) {
    const size_t comma = wanted.find(',', pos);
    std::string name =
        wanted.substr(pos, comma == std::string::npos ? comma : comma - pos);
    pos = comma == std::string::npos ? wanted.size() : comma + 1;
    bool known = false;
    for (const char* candidate : kAll) known = known || name == candidate;
    if (name.empty()) continue;
    if (known) {
      selected.insert(name);
    } else {
      unknown.push_back(name);
    }
  }
  if (!unknown.empty()) {
    std::fprintf(stderr, "tinynav_node: unknown TINYNAV_COMPONENTS entries:");
    for (const auto& name : unknown) std::fprintf(stderr, " %s", name.c_str());
    std::fprintf(stderr, " (valid: imu perception mapping planning)\n");
    return 1;
  }
  if (selected.empty()) {
    std::fprintf(stderr, "tinynav_node: TINYNAV_COMPONENTS selects nothing\n");
    return 1;
  }
  const auto want = [&selected](const char* name) { return selected.count(name) > 0; };

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  std::vector<rclcpp::Node::SharedPtr> nodes;
  if (want("imu")) nodes.push_back(make_imu_propagator(options));
  if (want("perception")) nodes.push_back(make_perception(options));
  if (want("mapping")) nodes.push_back(make_mapping(options));
  if (want("planning")) nodes.push_back(make_planning(options));
  for (const auto& node : nodes) executor.add_node(node);

  // Own logger: this line belongs to the host, not to imu_propagator's file.
  RCLCPP_INFO(rclcpp::get_logger("tinynav"), "tinynav_node: %zu components up (%s), intra-process comms on", nodes.size(), wanted.c_str());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
