// Port of reference/tinynav/core/logsetup.py for the C++ stack — the single
// place that knows the per-node log layout, the rotation rule and the
// retention rule, so components only decide *their logger name* (the rcutils
// hook maps it to the file tag). The .cpp carries the piece-by-piece mapping
// to the Python module; the ROS-facing surface is only install(), called
// once from main(). The sink and sweeper are plain C++ so tests run without
// a ROS graph.
#ifndef TINYNAV_CPP__LOGGING_SETUP_HPP_
#define TINYNAV_CPP__LOGGING_SETUP_HPP_

#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace tinynav {
namespace logging_setup {

/// Resolve the logs root exactly like logsetup.logs_root(): $TINYNAV_DB_PATH
/// when set non-empty, "logs" (cwd-relative — the tmux windows run at the
/// repo root, which is how both stacks share <repo>/logs) when set empty,
/// /tinynav/tinynav_db when unset.
std::string logs_root();

/// Port of logsetup.sweep: remove day-named directories under root strictly
/// older than retention_days (default 14 = logsetup.RETENTION_DAYS). Returns
/// the removed directory names.
std::vector<std::string> sweep(int retention_days, const std::string& root);

/// Install everything once from main(), after rclcpp::init(): the rcutils
/// output-handler hook that routes every RCLCPP_* line into its tag's
/// <root>/<day>/<tag>.log (DEBUG+ in files, like the Python file handlers),
/// the console copy (INFO+ on stderr, same format), the size fuse
/// (TINYNAV_LOG_MAX_MB, default 10, 0 disables), the 14-day sweeper thread
/// and the fd-level console tee into console.log (TINYNAV_LOG_CONSOLE=0
/// disables). Idempotent.
void install();

namespace detail {

/// Port of the Python tags: logger names carry a _node suffix in ROS
/// (perception_node → perception, map_node → map, planning_node → planning,
/// imu_propagator_node → imu_propagator, anything else as-is, empty →
/// tinynav).
std::string tag_for_logger(const std::string& logger_name);

/// Python _FORMAT '%(asctime)s.%(msecs)03d %(levelname)s %(name)s:
/// %(message)s' with _DATEFMT '%Y-%m-%dT%H:%M:%S' — local RFC3339 with
/// milliseconds. severity uses the rcutils severity constants.
std::string format_line(int severity, const std::string& tag,
                        long long stamp_ns, const std::string& message);

/// Port of logsetup.DayDirFileHandler plus the fleet's 10MB size fuse:
/// <root>/<YYYY-MM-DD>/<tag>.log, reopened when the local date rolls; when
/// the open file reaches max_bytes it shifts to .1/.2/.3.log (glog
/// semantics, <tag>.log always newest). max_bytes == 0 disables the fuse.
/// write() appends one line + '\n' and flushes, and never throws into the
/// caller (a bad record must not kill the node — the VIO-loop incident in
/// the logsetup docstring).
class DayDirFileSink {
  public:
    DayDirFileSink(std::string root, std::string tag, long long max_bytes);
    void write(long long stamp_ns, const std::string& line);

  private:
    bool open(const char* day);
    void rotate();

    std::string root_;
    std::string tag_;
    long long max_bytes_;
    std::string day_;
    long long written_ = 0;
    std::ofstream fh_;
    std::mutex mutex_;
};

constexpr int kDefaultRetentionDays = 14;   // logsetup.RETENTION_DAYS
constexpr int kSweepPeriodSec = 24 * 3600;  // logsetup.SWEEP_PERIOD_SEC

}  // namespace detail
}  // namespace logging_setup
}  // namespace tinynav

#endif  // TINYNAV_CPP__LOGGING_SETUP_HPP_
