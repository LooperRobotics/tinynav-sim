// Tests for the C++ port of reference/tinynav/core/logsetup.py — the day-dir
// sink, the size fuse, the retention sweep and the format contract. Runs
// without a ROS graph (the only ROS-facing piece, the rcutils hook, is
// exercised by the e2e rig runs).
#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

#include <rcutils/logging.h>

#include "tinynav_cpp/logging_setup.hpp"

namespace fs = std::filesystem;
namespace ls = tinynav::logging_setup;
namespace d = tinynav::logging_setup::detail;

namespace {

std::string day_string(std::time_t t) {
  std::tm local{};
  localtime_r(&t, &local);
  char buf[16];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d", &local);
  return buf;
}

std::string day_string_offset(int days) {
  return day_string(std::time(nullptr) + static_cast<std::time_t>(days) * 86400);
}

// A stamp at today's HH:MM:SS.mmm local time (mktime/localtime round trip,
// so the test is TZ-safe).
long long stamp_ns_for(int hour, int min, int sec, int millis) {
  std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  local.tm_hour = hour;
  local.tm_min = min;
  local.tm_sec = sec;
  const std::time_t t = std::mktime(&local);
  return static_cast<long long>(t) * 1000000000LL +
         static_cast<long long>(millis) * 1000000LL;
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

class TempDir {
  public:
    TempDir()
        : name_("tinynav_log_test_" + std::to_string(::getpid()) + "-" +
                std::to_string(s_counter++)),
          path_(fs::temp_directory_path() / name_),
          path_str_(path_.string()) {}
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    const std::string& str() const { return path_str_; }

  private:
    static int s_counter;
    std::string name_;
    fs::path path_;
    std::string path_str_;
};
int TempDir::s_counter = 0;

}  // namespace

TEST(logging_setup, tag_for_logger_strips_node_suffix) {
  EXPECT_EQ(d::tag_for_logger("perception_node"), "perception");
  EXPECT_EQ(d::tag_for_logger("map_node"), "map");
  EXPECT_EQ(d::tag_for_logger("planning_node"), "planning");
  EXPECT_EQ(d::tag_for_logger("imu_propagator_node"), "imu_propagator");
  EXPECT_EQ(d::tag_for_logger("tinynav"), "tinynav");
  EXPECT_EQ(d::tag_for_logger(""), "tinynav");
}

TEST(logging_setup, format_line_matches_python_layout) {
  // Python _FORMAT '%(asctime)s.%(msecs)03d %(levelname)s %(name)s:
  // %(message)s' with _DATEFMT '%Y-%m-%dT%H:%M:%S'.
  const long long stamp = stamp_ns_for(10, 11, 22, 123);
  const std::string line = d::format_line(RCUTILS_LOG_SEVERITY_INFO, "map", stamp, "hello");
  EXPECT_NE(line.find("T10:11:22.123 INFO map: hello"), std::string::npos) << line;

  EXPECT_NE(d::format_line(RCUTILS_LOG_SEVERITY_DEBUG, "map", stamp, "x").find(" DEBUG "), std::string::npos);
  EXPECT_NE(d::format_line(RCUTILS_LOG_SEVERITY_WARN, "map", stamp, "x").find(" WARNING "), std::string::npos);
  EXPECT_NE(d::format_line(RCUTILS_LOG_SEVERITY_ERROR, "map", stamp, "x").find(" ERROR "), std::string::npos);
  EXPECT_NE(d::format_line(RCUTILS_LOG_SEVERITY_FATAL, "map", stamp, "x").find(" CRITICAL "), std::string::npos);
}

TEST(logging_setup, day_dir_sink_writes_python_layout) {
  TempDir root;
  d::DayDirFileSink sink(root.str(), "map", 0);
  const long long stamp = stamp_ns_for(10, 11, 22, 123);
  sink.write(stamp, d::format_line(RCUTILS_LOG_SEVERITY_INFO, "map", stamp, "hello"));

  const fs::path path = fs::path(root.str()) / day_string(stamp / 1000000000LL) / "map.log";
  ASSERT_TRUE(fs::exists(path)) << path;
  EXPECT_NE(read_file(path).find("T10:11:22.123 INFO map: hello\n"), std::string::npos);
}

TEST(logging_setup, day_dir_sink_appends_across_instances) {
  // A second process run on the same day appends (and the fuse counts the
  // pre-existing bytes).
  TempDir root;
  const long long stamp = stamp_ns_for(10, 11, 22, 123);
  {
    d::DayDirFileSink sink(root.str(), "map", 0);
    sink.write(stamp, "first");
  }
  {
    d::DayDirFileSink sink(root.str(), "map", 0);
    sink.write(stamp, "second");
  }
  const fs::path path = fs::path(root.str()) / day_string(stamp / 1000000000LL) / "map.log";
  EXPECT_NE(read_file(path).find("first\n"), std::string::npos);
  EXPECT_NE(read_file(path).find("second\n"), std::string::npos);
}

TEST(logging_setup, day_dir_sink_size_fuse_shifts_chain) {
  TempDir root;
  d::DayDirFileSink sink(root.str(), "map", 64);  // tiny fuse for the test
  const long long stamp = stamp_ns_for(10, 11, 22, 123);
  // Each line ~35 bytes: L0,L1 fill the file; L2 trips the fuse, ...
  for (int i = 0; i < 5; ++i) {
    sink.write(stamp, "line " + std::to_string(i) + " aaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  }
  const fs::path base = fs::path(root.str()) / day_string(stamp / 1000000000LL) / "map";
  // L0,L1 -> .2.log; L2,L3 -> .1.log; L4 -> the always-newest map.log.
  const std::string b2 = read_file(base.string() + ".2.log");
  const std::string b1 = read_file(base.string() + ".1.log");
  const std::string b0 = read_file(base.string() + ".log");
  EXPECT_NE(b2.find("line 0 "), std::string::npos);
  EXPECT_NE(b2.find("line 1 "), std::string::npos);
  EXPECT_EQ(b2.find("line 2 "), std::string::npos);
  EXPECT_NE(b1.find("line 2 "), std::string::npos);
  EXPECT_NE(b1.find("line 3 "), std::string::npos);
  EXPECT_NE(b0.find("line 4 "), std::string::npos);
  EXPECT_EQ(b0.find("line 3 "), std::string::npos);
  EXPECT_FALSE(fs::exists(base.string() + ".3.log"));
}

TEST(logging_setup, sweep_removes_only_old_day_dirs) {
  TempDir root;
  const std::string old_day = day_string_offset(-30);   // beyond retention
  const std::string recent_day = day_string_offset(-2);  // inside retention
  const std::string today = day_string_offset(0);
  fs::create_directories(fs::path(root.str()) / old_day);
  fs::create_directories(fs::path(root.str()) / recent_day);
  fs::create_directories(fs::path(root.str()) / today);
  fs::create_directories(fs::path(root.str()) / "random_name");  // not a day
  { std::ofstream f(fs::path(root.str()) / "2026-08-02"); f << "a file, not a dir"; }

  const auto removed = ls::sweep(d::kDefaultRetentionDays, root.str());

  ASSERT_EQ(removed.size(), 1u) << "removed: " << [&] {
    std::string s;
    for (const auto& r : removed) s += " " + r;
    return s;
  }();
  EXPECT_EQ(removed[0], old_day);
  EXPECT_FALSE(fs::exists(fs::path(root.str()) / old_day));
  EXPECT_TRUE(fs::exists(fs::path(root.str()) / recent_day));
  EXPECT_TRUE(fs::exists(fs::path(root.str()) / today));
  EXPECT_TRUE(fs::exists(fs::path(root.str()) / "random_name"));
  EXPECT_TRUE(fs::exists(fs::path(root.str()) / "2026-08-02"));
}

TEST(logging_setup, sweep_missing_root_is_noop) {
  TempDir root;  // never created on disk
  const auto removed = ls::sweep(d::kDefaultRetentionDays, root.str());
  EXPECT_TRUE(removed.empty());
}
