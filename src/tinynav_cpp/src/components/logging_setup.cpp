// Port of reference/tinynav/core/logsetup.py for the C++ stack. The Python
// module is the spec; piece-by-piece:
//
//   DayDirFileHandler -> detail::DayDirFileSink (plus the 10MB size fuse the
//                        fleet added — see the QA knowledge base 方向六 6.5:
//                        perception produced 4.2GB/day before it)
//   sweep             -> sweep() (day directories, name-string cutoff)
//   setup_logging     -> install(): instead of a stdlib logger per node, ONE
//                        rcutils output-handler hook routes every RCLCPP_*
//                        line by logger name, so the component call sites
//                        stay untouched (the pilot-side get_logger() shim
//                        solved the same problem from the other end)
//   capture_console   -> capture_console(): fd-level tee of stdout+stderr
//                        into <root>/<day>/console.log
//
// Retention is the same single number (RETENTION_DAYS = 14), swept at start
// and then every SWEEP_PERIOD_SEC by a daemon thread.
//
// Why a hand-rolled sink rather than spdlog: its daily_file_sink is final,
// and neither stock sink can express the per-day directory layout plus the
// size fuse — this sink IS the reference handler, ported. spdlog is present
// in both build images but would only have contributed a mutex wrapper.
#include "tinynav_cpp/logging_setup.hpp"

#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <atomic>
#include <filesystem>
#include <memory>
#include <regex>
#include <system_error>
#include <thread>
#include <unordered_map>

#include <rcutils/logging.h>

namespace tinynav {
namespace logging_setup {
namespace {

std::string g_logs_root;
long long g_max_bytes = 10LL * 1024 * 1024;
std::atomic<long long> g_dropped_lines{0};

std::mutex g_sinks_mutex;
std::unordered_map<std::string, std::shared_ptr<detail::DayDirFileSink>> g_sinks;

std::shared_ptr<detail::DayDirFileSink> sink_for(const std::string& tag) {
    std::lock_guard<std::mutex> lock(g_sinks_mutex);
    const auto it = g_sinks.find(tag);
    if (it != g_sinks.end()) return it->second;
    auto sink = std::make_shared<detail::DayDirFileSink>(g_logs_root, tag, g_max_bytes);
    g_sinks.emplace(tag, sink);
    return sink;
}

// The rcutils hook. The default console handler is REPLACED, so this both
// formats and prints; the console copy is INFO+ (the Python
// setup_logging(console_level=INFO) contract) and files take DEBUG+ (the
// Python logger runs at DEBUG).
void rcutils_hook(const rcutils_log_location_t* /*location*/, int severity,
                  const char* name, rcutils_time_point_value_t stamp,
                  const char* format, va_list* args) {
    try {
        char stackbuf[4096];
        va_list dup_args;
        va_copy(dup_args, *args);
        const int n = std::vsnprintf(stackbuf, sizeof(stackbuf), format, dup_args);
        va_end(dup_args);
        if (n < 0) return;
        std::string message;
        if (static_cast<size_t>(n) < sizeof(stackbuf)) {
            message.assign(stackbuf, static_cast<size_t>(n));
        } else {
            message.resize(static_cast<size_t>(n) + 1);
            va_copy(dup_args, *args);
            std::vsnprintf(&message[0], message.size(), format, dup_args);
            va_end(dup_args);
            message.resize(static_cast<size_t>(n));
        }
        const std::string tag = detail::tag_for_logger(name != nullptr ? name : "tinynav");
        const std::string line = detail::format_line(severity, tag, stamp, message);
        if (severity >= RCUTILS_LOG_SEVERITY_INFO) {
            std::fwrite(line.data(), 1, line.size(), stderr);
            std::fputc('\n', stderr);
            std::fflush(stderr);
        }
        sink_for(tag)->write(stamp, line);
    } catch (...) {
        // Mirror of the DayDirFileHandler contract: a bad record is dropped,
        // never propagated into the caller's (VIO) loop.
        g_dropped_lines.fetch_add(1, std::memory_order_relaxed);
    }
}

// Port of logsetup.capture_console: stdout+stderr both point at a pipe; a
// pump thread forwards every byte to the real stdout and appends it to
// <root>/<day>/console.log — the raw mirror that also catches bare prints
// and pre-logging bootstrap output. The pump must never die: the fleet
// incident was a dead pump leaving the pipe full, which wedges every thread
// that logs.
bool capture_console(const std::string& root) {
    const int real_stdout = ::dup(STDOUT_FILENO);
    int fds[2];
    if (::pipe(fds) != 0) {
        ::close(real_stdout);
        return false;
    }
    ::dup2(fds[1], STDOUT_FILENO);
    ::dup2(fds[1], STDERR_FILENO);
    ::close(fds[1]);

    // stdout is a pipe now — libc would fully buffer it. Unbuffered matches
    // the fleet's PYTHONUNBUFFERED=1 rule (QA 方向六 6.1: buffered logs once
    // lagged the scene by 8 minutes).
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::thread([read_fd = fds[0], out = real_stdout, root]() {
        std::string day;
        std::ofstream fh;
        for (;;) {
            char buf[65536];
            const ssize_t n = ::read(read_fd, buf, sizeof(buf));
            if (n < 0) {
                if (errno == EINTR) continue;
                return;
            }
            if (n == 0) return;  // write end closed: process is exiting
            for (ssize_t off = 0; off < n;) {
                const ssize_t w = ::write(out, buf + off, static_cast<size_t>(n - off));
                if (w <= 0) return;
                off += w;
            }
            try {
                const std::time_t now = std::time(nullptr);
                std::tm local{};
                localtime_r(&now, &local);
                char today[16];
                std::strftime(today, sizeof(today), "%Y-%m-%d", &local);
                if (!fh.is_open() || day != today) {
                    std::error_code ec;
                    std::filesystem::create_directories(root + "/" + today, ec);
                    if (!ec) {
                        fh.open(root + "/" + today + "/console.log",
                                std::ios::app | std::ios::binary);
                        day = today;
                    }
                }
                if (fh.is_open()) {
                    fh.write(buf, n);
                    fh.flush();
                }
            } catch (...) {
                // The file copy is best-effort; the passthrough above is the
                // contract. A failure here must not stop the pump.
            }
        }
    }).detach();
    return true;
}

}  // namespace

std::string logs_root() {
    // os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db') + '/logs',
    // including the join('' , 'logs') == 'logs' quirk when the variable is
    // exported empty (run_simulator.sh exports it in every window).
    const char* db = std::getenv("TINYNAV_DB_PATH");
    if (db == nullptr) return "/tinynav/tinynav_db/logs";
    if (db[0] == '\0') return "logs";
    return std::string(db) + "/logs";
}

std::vector<std::string> sweep(int retention_days, const std::string& root) {
    std::vector<std::string> removed;
    const std::time_t cutoff = std::time(nullptr) -
                               static_cast<std::time_t>(retention_days) * 86400;
    std::tm cutoff_tm{};
    localtime_r(&cutoff, &cutoff_tm);
    char cutoff_buf[16];
    std::strftime(cutoff_buf, sizeof(cutoff_buf), "%Y-%m-%d", &cutoff_tm);

    std::error_code ec;
    std::filesystem::directory_iterator it(root, ec), end;
    if (ec) return removed;
    static const std::regex day_re(R"(^\d{4}-\d{2}-\d{2}$)");
    for (; it != end; ++it) {
        const std::string name = it->path().filename().string();
        if (!std::regex_match(name, day_re) || name >= cutoff_buf) continue;
        if (!it->is_directory(ec)) continue;  // rmtree never removed plain files
        std::filesystem::remove_all(it->path(), ec);
        removed.push_back(name);
    }
    return removed;
}

void install() {
    static std::atomic<bool> installed{false};
    bool expected = false;
    if (!installed.compare_exchange_strong(expected, true)) return;

    // Tee first, so even the setup lines below land in console.log.
    const bool console_enabled = [] {
        const char* env = std::getenv("TINYNAV_LOG_CONSOLE");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    g_logs_root = logs_root();
    if (console_enabled) capture_console(g_logs_root);

    // TINYNAV_LOG_MAX_MB: the fleet's size-fuse knob; <=0 disables.
    if (const char* env = std::getenv("TINYNAV_LOG_MAX_MB")) {
        const long long parsed = std::atoll(env);
        g_max_bytes = parsed > 0 ? parsed * 1024 * 1024 : 0;
    }

    // DEBUG must reach the hook so files keep the Python file handlers'
    // debug-through level; the hook re-filters the console copy to INFO+.
    for (const char* logger : {"perception_node", "map_node", "planning_node",
                               "imu_propagator_node", "tinynav"}) {
        (void)rcutils_logging_set_logger_level(logger, RCUTILS_LOG_SEVERITY_DEBUG);
    }
    rcutils_logging_set_output_handler(&rcutils_hook);

    // Retention sweeper: at start, then every SWEEP_PERIOD_SEC — a rig runs
    // for months (logsetup._ensure_sweeper).
    std::thread([] {
        for (;;) {
            const auto removed = sweep(detail::kDefaultRetentionDays, g_logs_root);
            if (!removed.empty()) {
                std::string names;
                for (const auto& r : removed) {
                    names += ' ';
                    names += r;
                }
                std::fprintf(stderr, "log retention: removed %zu day dirs:%s\n",
                             removed.size(), names.c_str());
                std::fflush(stderr);
            }
            std::this_thread::sleep_for(std::chrono::seconds(detail::kSweepPeriodSec));
        }
    }).detach();
}

namespace detail {

std::string tag_for_logger(const std::string& logger_name) {
    std::string tag = logger_name.empty() ? "tinynav" : logger_name;
    if (tag.size() > 5 && tag.compare(tag.size() - 5, 5, "_node") == 0) {
        tag.resize(tag.size() - 5);
    }
    return tag;
}

const char* level_name(int severity) {
    switch (severity) {
        case RCUTILS_LOG_SEVERITY_DEBUG: return "DEBUG";
        case RCUTILS_LOG_SEVERITY_INFO: return "INFO";
        case RCUTILS_LOG_SEVERITY_WARN: return "WARNING";
        case RCUTILS_LOG_SEVERITY_ERROR: return "ERROR";
        default: return "CRITICAL";  // FATAL — python logging's top name
    }
}

std::string format_line(int severity, const std::string& tag, long long stamp_ns,
                        const std::string& message) {
    const std::time_t seconds = static_cast<std::time_t>(stamp_ns / 1'000'000'000LL);
    const long long millis = (stamp_ns % 1'000'000'000LL) / 1'000'000LL;
    std::tm local{};
    localtime_r(&seconds, &local);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", &local);
    char head[64];
    std::snprintf(head, sizeof(head), ".%03lld %s %s: ", millis,
                  level_name(severity), tag.c_str());
    std::string line;
    line.reserve(std::strlen(stamp) + std::strlen(head) + message.size() + 1);
    line += stamp;
    line += head;
    line += message;
    return line;
}

DayDirFileSink::DayDirFileSink(std::string root, std::string tag, long long max_bytes)
    : root_(std::move(root)), tag_(std::move(tag)), max_bytes_(max_bytes) {}

void DayDirFileSink::write(long long stamp_ns, const std::string& line) {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::time_t seconds = static_cast<std::time_t>(stamp_ns / 1'000'000'000LL);
        std::tm local{};
        localtime_r(&seconds, &local);
        char day[16];
        std::strftime(day, sizeof(day), "%Y-%m-%d", &local);
        if (!fh_.is_open() || day_ != day) {
            if (!open(day)) return;
        } else if (max_bytes_ > 0 && written_ >= max_bytes_) {
            rotate();
        }
        fh_ << line << '\n';
        fh_.flush();
        written_ += static_cast<long long>(line.size()) + 1;
    } catch (...) {
        // See the logsetup docstring: emit() must degrade, never propagate —
        // the unguarded first version let one bad debug() kill the VIO loop.
        g_dropped_lines.fetch_add(1, std::memory_order_relaxed);
    }
}

bool DayDirFileSink::open(const char* day) {
    fh_.close();
    std::error_code ec;
    std::filesystem::create_directories(root_ + "/" + day, ec);
    if (ec) return false;
    const std::string path = root_ + "/" + day + "/" + tag_ + ".log";
    fh_.open(path, std::ios::app | std::ios::binary);
    if (!fh_.is_open()) return false;
    day_ = day;
    // Pre-existing content (an earlier process run appended to today's file)
    // counts toward the fuse threshold.
    const auto size = std::filesystem::file_size(path, ec);
    written_ = ec ? 0 : static_cast<long long>(size);
    return true;
}

void DayDirFileSink::rotate() {
    // glog-style shift: .2 -> .3, .1 -> .2, current -> .1, then reopen a
    // fresh <tag>.log — the newest content is always in <tag>.log.
    fh_.close();
    const std::string base = root_ + "/" + day_ + "/" + tag_;
    std::error_code ec;
    std::filesystem::remove(base + ".3.log", ec);
    ec.clear();
    for (int i = 2; i >= 1; --i) {
        std::filesystem::rename(base + "." + std::to_string(i) + ".log",
                                base + "." + std::to_string(i + 1) + ".log", ec);
        ec.clear();
    }
    std::filesystem::rename(base + ".log", base + ".1.log", ec);
    open(day_.c_str());
}

}  // namespace detail
}  // namespace logging_setup
}  // namespace tinynav
