// Port of tinynav-pilot/tinynav/platforms/unitree_control.py — the real-dog
// chassis bridge: rt/cmd_vel (SDK channel) drives the SportClient Move/gait
// FSM, rt/sportmodestate feeds the chassis watchdog, rt/lowstate reports the
// battery, rt/utlidar/robot_odom is republished as /unitree/odometry, and
// rt/service/command plays the sit/stand action sequences.
//
// Deliberate differences vs the python (this is the CPU fix — the python was
// 62-74% of a core, dominated by IDL deserialization of two ~500Hz streams):
//   - /battery is published at 1 Hz (python: every lowstate message, ~500 Hz).
//   - rt/sportmodestate is processed at 10 Hz (python: every message, ~500 Hz).
//     Watchdog timescales (0.5 s check, 1 s silence) are far coarser than the
//     throttle, so semantics are unchanged.
//   - rt/utlidar/robot_odom keeps the python 50 Hz cap.
// Everything else (gait FSM, watchdog thresholds, action sequences, status
// semantics — /robot_status is claimed only when every SDK call returned 0)
// follows the python line by line.
//
// The node builds only when UNITREE_SDK2_ROOT points at a unitree_sdk2
// checkout (prebuilt libunitree_sdk2.a per arch); see CMakeLists.txt.
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/b2/sport/sport_client.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/ros2/Twist_.hpp>
#include <unitree/idl/ros2/Odometry_.hpp>
#include <unitree/idl/ros2/String_.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>

#include "tinynav_cpp/logging_setup.hpp"

namespace {

constexpr double kSlowRpcS = 0.3;            // a reply RPC slower than this is logged
constexpr double kCmdGapS = 0.5;             // rt/cmd_vel pause that counts as a gap
constexpr double kStallCmdV = 0.1;           // chassis watchdog thresholds
constexpr double kStallCmdW = 0.2;
constexpr double kStallChassisV = 0.03;
constexpr double kStallChassisW = 0.05;
constexpr double kStallAfterS = 2.0;
constexpr double kRepeatS = 5.0;
constexpr double kSportStateSilentS = 1.0;
// The two throttles (see the header comment).
constexpr double kBatteryPeriodS = 1.0;      // battery: 1 Hz (python: ~500 Hz)
constexpr double kSportStatePeriodS = 0.1;   // sportmodestate: 10 Hz (python: ~500 Hz)
constexpr double kChassisOdomPeriodS = 0.02; // 50 Hz cap, same as the python

double monotonic_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const char* robot_status_string(bool standing) { return standing ? "standup" : "sitting"; }

// Per-model bound SDK calls (the python duck-types one sport_client across
// go2/b2/g1 clients; C++ gets one bundle of std::functions instead). Empty
// call = the model has no such RPC and the caller must skip it.
struct SportCalls {
    std::function<int32_t()> classic_walk;
    std::function<int32_t()> stand_up;
    std::function<int32_t()> stand_down;
    std::function<int32_t()> balance_stand;
    std::function<int32_t()> switch_gait;  // b2/b2w only
    std::function<int32_t(float, float, float)> move;
    // g1-only FSM ids for the sit/stand sequence (empty for quadrupeds).
    std::function<int32_t()> damp;
    std::function<int32_t()> squat2stand_up;
    std::function<int32_t()> stand_up2squat;
};

// Keeps the concrete SDK client alive; unitree clients are not polymorphic.
struct SportClientHolder {
    std::unique_ptr<unitree::robot::go2::SportClient> go2;
    std::unique_ptr<unitree::robot::b2::SportClient> b2;
    std::unique_ptr<unitree::robot::g1::LocoClient> g1;
};

class GaitWorker {
  public:
    GaitWorker(std::function<int32_t()> call, rclcpp::Logger log,
               std::string name = "ClassicWalk")
        : call_(std::move(call)), log_(log), name_(std::move(name)),
          thread_([this] { run(); }) {}

    // Non-blocking: safe to call from a reader thread.
    void request() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            wake_ = true;
        }
        cv_.notify_one();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            wake_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    // One gait assertion, timed and logged. Separated out so it can be driven
    // without a thread.
    void run_once() {
        const double t0 = monotonic_s();
        int32_t code = 0;
        try {
            code = call_();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(log_, "[sport] %s raised: %s", name_.c_str(), e.what());
            return;
        }
        const double took = monotonic_s() - t0;
        if (code != 0 || took > kSlowRpcS) {
            RCLCPP_WARN(log_, "[sport] %s code=%d took %.2fs", name_.c_str(), code, took);
        }
    }

  private:
    void run() {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return wake_; });
            if (stop_) {
                return;
            }
            wake_ = false;
            lock.unlock();
            run_once();
        }
    }

    std::function<int32_t()> call_;
    rclcpp::Logger log_;
    std::string name_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool wake_ = false;
    bool stop_ = false;
    std::thread thread_;
};

// Compares what was commanded with what rt/sportmodestate says the chassis is
// doing. Logs only; never changes a command. The python relied on the GIL to
// serialize on_cmd/on_sport_state/check from three threads; here: one mutex.
class ChassisWatch {
  public:
    explicit ChassisWatch(rclcpp::Logger log) : log_(log) {}

    void on_cmd(double now, double vx, double vy, double wz) {
        std::lock_guard<std::mutex> lock(mutex_);
        cmd_ = {vx, vy, wz};
        cmd_at_ = now;
    }

    void on_sport_state(double now, uint8_t mode, uint8_t gait, uint32_t error_code,
                        double vx, double vy, double yaw_speed,
                        const std::array<float, 4>& range_obstacle) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool have_state = state_.has_value();
        if (!have_state || state_->mode != mode || state_->gait != gait ||
            state_->error_code != error_code) {
            const std::string was =
                have_state ? " (was mode=" + std::to_string(state_->mode) +
                                 " gait=" + std::to_string(state_->gait) + " error_code=" +
                                 std::to_string(state_->error_code) + ")"
                           : "";
            if (error_code != 0) {
                RCLCPP_WARN(log_, "[chassis] mode=%u gait=%u error_code=%u%s", mode, gait,
                            error_code, was.c_str());
            } else {
                RCLCPP_INFO(log_, "[chassis] mode=%u gait=%u error_code=%u%s", mode, gait,
                            error_code, was.c_str());
            }
            state_ = State{mode, gait, error_code};
        }
        v_ = {vx, vy};
        yaw_speed_ = yaw_speed;
        range_obstacle_ = range_obstacle;
        state_at_ = now;
    }

    void check(double now) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_at_.has_value()) {
            const double silent = now - *state_at_;
            if (silent > kSportStateSilentS && !silent_logged_) {
                RCLCPP_WARN(log_, "[chassis] rt/sportmodestate silent for %.1fs", silent);
                silent_logged_ = true;
            } else if (silent <= kSportStateSilentS && silent_logged_) {
                RCLCPP_INFO(log_, "[chassis] rt/sportmodestate back");
                silent_logged_ = false;
            }
        }

        const auto [vx, vy, wz] = cmd_;
        const bool commanded =
            cmd_at_.has_value() && now - *cmd_at_ < kCmdGapS &&
            (std::hypot(vx, vy) >= kStallCmdV || std::abs(wz) >= kStallCmdW);
        const bool still =
            state_at_.has_value() && std::hypot(v_[0], v_[1]) < kStallChassisV &&
            std::abs(yaw_speed_) < kStallChassisW;
        if (commanded && still) {
            if (!stall_since_.has_value()) {
                stall_since_ = now;
            }
            const double held = now - *stall_since_;
            if (held >= kStallAfterS &&
                (!stall_logged_at_.has_value() || now - *stall_logged_at_ >= kRepeatS)) {
                stall_logged_at_ = now;
                std::string mode_gait_err = "none";
                if (state_.has_value()) {
                    mode_gait_err = "mode=" + std::to_string(state_->mode) +
                                    " gait=" + std::to_string(state_->gait) +
                                    " error_code=" + std::to_string(state_->error_code);
                }
                std::string obstacles = "[";
                for (size_t i = 0; i < range_obstacle_.size(); ++i) {
                    obstacles += std::to_string(range_obstacle_[i]);
                    obstacles += (i + 1 < range_obstacle_.size()) ? ", " : "]";
                }
                RCLCPP_WARN(log_,
                            "[chassis] not executing for %.1fs: commanded vx=%.2f vy=%.2f "
                            "wz=%.2f, chassis v=%.3f yaw_speed=%.3f %s range_obstacle=%s",
                            held, vx, vy, wz, std::hypot(v_[0], v_[1]), yaw_speed_,
                            mode_gait_err.c_str(), obstacles.c_str());
            }
        } else {
            if (stall_logged_at_.has_value()) {
                RCLCPP_INFO(log_, "[chassis] executing again after %.1fs",
                            now - stall_since_.value_or(now));
            }
            stall_since_.reset();
            stall_logged_at_.reset();
        }
    }

  private:
    struct State {
        uint8_t mode;
        uint8_t gait;
        uint32_t error_code;
    };

    rclcpp::Logger log_;
    std::mutex mutex_;
    std::array<double, 3> cmd_ = {0.0, 0.0, 0.0};
    std::optional<double> cmd_at_;
    std::optional<State> state_;
    std::array<double, 2> v_ = {0.0, 0.0};
    double yaw_speed_ = 0.0;
    std::array<float, 4> range_obstacle_ = {0.0f, 0.0f, 0.0f, 0.0f};
    std::optional<double> state_at_;
    std::optional<double> stall_since_;
    std::optional<double> stall_logged_at_;
    bool silent_logged_ = false;
};

// true when the model reuses the go2/b2 quadruped SportClient surface.
bool is_quadruped_model(const std::string& model) {
    return model == "go2" || model == "go2w" || model == "b2" || model == "b2w";
}

}  // namespace

class UnitreeControlNode : public rclcpp::Node {
  public:
    UnitreeControlNode(const rclcpp::NodeOptions& options) : Node("ros2_unitree_manager", options) {
        network_interface_ = declare_parameter<std::string>("network_interface", "enP8p1s0");
        // Fleet parity: the python read ROBOT_TYPE from the environment and
        // refused to start without it. The parameter keeps that required
        // choice explicit; the env remains the default when unset.
        const char* env_model = std::getenv("ROBOT_TYPE");
        const std::string env_default = env_model != nullptr ? env_model : "go2";
        robot_model_ = declare_parameter<std::string>("robot_model", env_default);
        for (char& c : robot_model_) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        is_quadruped_ = is_quadruped_model(robot_model_);
        has_switch_gait_ = robot_model_ == "b2" || robot_model_ == "b2w";
        const bool supported =
            is_quadruped_ || robot_model_ == "g1";
        if (!supported) {
            RCLCPP_FATAL(get_logger(), "Unsupported robot model: '%s' (go2 go2w b2 b2w g1)",
                         robot_model_.c_str());
            throw std::runtime_error("unsupported robot model " + robot_model_);
        }

        RCLCPP_INFO(get_logger(), "robot=%s interface=%s", robot_model_.c_str(),
                    network_interface_.c_str());
        unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);

        build_sport_client();
        sport_client_->SetTimeout(10.0f);
        sport_client_->Init();
        if (is_quadruped_) {
            sport_calls_.classic_walk();
        }
        robot_status_ = false;  // SITTING
        // The last command was non-zero (so the next non-zero is not a start).
        walking_ = false;
        // The next Move re-asserts ClassicWalk first: set at every motion start.
        gait_due_ = false;

        if (is_quadruped_) {
            gait_worker_ = std::make_unique<GaitWorker>(
                [this] { return sport_calls_.classic_walk(); }, get_logger());
        }

        twist_subscriber_ = std::make_unique<unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Twist_>>(
            "rt/cmd_vel");
        twist_subscriber_->InitChannel(
            [this](const void* msg) { TwistMessageHandler(*static_cast<const geometry_msgs::msg::dds_::Twist_*>(msg)); },
            10);

        action_subscriber_ =
            std::make_unique<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>>(
                "rt/service/command");
        action_subscriber_->InitChannel(
            [this](const void* msg) {
                ActionMessageHandler(*static_cast<const std_msgs::msg::dds_::String_*>(msg));
            },
            10);

        // g1's lowstate has no battery field; the handler skips battery
        // reporting for it (the subscription stays, like the python's).
        lowstate_subscriber_ = std::make_unique<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
            "rt/lowstate");
        lowstate_subscriber_->InitChannel(
            [this](const void* msg) {
                LowStateMessageHandler(*static_cast<const unitree_go::msg::dds_::LowState_*>(msg));
            },
            10);

        publisher_battery_ = create_publisher<std_msgs::msg::Float32>("/battery", 10);
        publisher_robot_status_ = create_publisher<std_msgs::msg::String>("/robot_status", 10);

        if (is_quadruped_) {
            publisher_chassis_odom_ =
                create_publisher<nav_msgs::msg::Odometry>("/unitree/odometry", 10);
            chassis_odom_subscriber_ =
                std::make_unique<unitree::robot::ChannelSubscriber<nav_msgs::msg::dds_::Odometry_>>(
                    "rt/utlidar/robot_odom");
            chassis_odom_subscriber_->InitChannel(
                [this](const void* msg) {
                    ChassisOdomMessageHandler(*static_cast<const nav_msgs::msg::dds_::Odometry_*>(msg));
                },
                10);

            sport_state_subscriber_ =
                std::make_unique<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(
                    "rt/sportmodestate");
            sport_state_subscriber_->InitChannel(
                [this](const void* msg) {
                    SportStateMessageHandler(*static_cast<const unitree_go::msg::dds_::SportModeState_*>(msg));
                },
                10);
            watch_timer_ = create_wall_timer(
                std::chrono::milliseconds(500),
                [this] { watch_.check(monotonic_s()); });
        }

        status_timer_ = create_wall_timer(std::chrono::seconds(1),
                                          [this] { publish_robot_status(); });
        RCLCPP_INFO(get_logger(), "ros2_unitree_manager up (battery 1Hz, sportmodestate 10Hz)");
    }

    ~UnitreeControlNode() override {
        if (gait_worker_ != nullptr) {
            gait_worker_->stop();
        }
    }

  private:
    // --- rt/cmd_vel: the Move path (paces the reader thread with 20 ms, like
    // the python; Move at planning rate must not hammer the sport service) --
    void TwistMessageHandler(const geometry_msgs::msg::dds_::Twist_& msg) {
        try {
            on_twist(monotonic_s(), static_cast<double>(msg.linear().x()),
                     static_cast<double>(msg.linear().y()),
                     static_cast<double>(msg.angular().z()));
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "cmd_vel handling failed: %s", e.what());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    void on_twist(double now, double vx, double vy, double wz) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        const std::optional<double> gap =
            last_twist_time_.has_value() ? std::optional<double>(now - *last_twist_time_)
                                         : std::nullopt;
        last_twist_time_ = now;
        if (gap.has_value() && *gap > kCmdGapS) {
            if (walking_) {
                RCLCPP_WARN(get_logger(), "[cmd_vel] rt/cmd_vel silent for %.2fs mid-motion", *gap);
            }
            gait_due_ = true;
        }

        if (vx != 0 || vy != 0 || wz != 0) {
            if (!walking_) {
                gait_due_ = true;
            }
            // Handed off, never called here: it is a reply RPC, and this thread must
            // stay free to keep pushing Move at the chassis.
            if (gait_due_ && gait_worker_ != nullptr) {
                gait_worker_->request();
            }
            gait_due_ = false;
            walking_ = true;
        } else {
            // A zero Move, not StopMove: StopMove is a reply RPC on this thread.
            walking_ = false;
        }
        const int32_t code = sport_calls_.move(vx, vy, wz);
        if (code != 0) {
            move_failed(now, code);
        }
        watch_.on_cmd(now, vx, vy, wz);
    }

    void move_failed(double now, int32_t code) {
        ++move_failures_;
        if (!move_failure_logged_at_.has_value() || now - *move_failure_logged_at_ >= kRepeatS) {
            RCLCPP_WARN(get_logger(), "[sport] Move send failed code=%d (%d failures so far)",
                        code, move_failures_);
            move_failure_logged_at_ = now;
        }
    }

    // --- rt/sportmodestate: watchdog state, throttled to 10 Hz -------------
    void SportStateMessageHandler(const unitree_go::msg::dds_::SportModeState_& msg) {
        const double now = monotonic_s();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (last_sport_state_time_.has_value() &&
                now - *last_sport_state_time_ < kSportStatePeriodS) {
                return;
            }
            last_sport_state_time_ = now;
        }
        // Runs on the SDK reader thread: keep it to field copies.
        watch_.on_sport_state(now, msg.mode(), msg.gait_type(), msg.error_code(),
                              msg.velocity()[0], msg.velocity()[1], msg.yaw_speed(),
                              msg.range_obstacle());
    }

    // --- rt/service/command: sit/stand action sequences --------------------
    void ActionMessageHandler(const std_msgs::msg::dds_::String_& msg) {
        RCLCPP_INFO(get_logger(), "ActionMessageHandler received: '%s'", msg.data().c_str());
        // unitree_sdk2py's reader thread calls this with no except around it, so an
        // exception escaping here kills that thread and the subscription goes deaf
        // for the rest of the run -- every later sit/stand silently dropped, while
        // the process still looks healthy. One bad action must not cost the channel.
        try {
            play_action(msg.data());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "action failed: %s", e.what());
        }
    }

    void play_action(const std::string& data) {
        const size_t space = data.find(' ');
        if (space == std::string::npos || data.substr(0, space) != "play") {
            return;
        }
        const std::string action_key = data.substr(space + 1, data.find(' ', space + 1) - (space + 1));
        if (action_key == "sit") {
            if (is_quadruped_) {
                play_steps("Sitting", {{"StandDown", sport_calls_.stand_down}}, false);
            } else {
                play_steps("Sitting", {{"StandUp2Squat", sport_calls_.stand_up2squat}}, false);
            }
        } else if (action_key == "stand") {
            if (is_quadruped_) {
                std::vector<std::pair<std::string, std::function<int32_t()>>> steps = {
                    {"StandUp", sport_calls_.stand_up},
                    {"BalanceStand", sport_calls_.balance_stand},
                    {"ClassicWalk", sport_calls_.classic_walk}};
                if (has_switch_gait_) {
                    steps.emplace_back("SwitchGait", sport_calls_.switch_gait);
                }
                play_steps("Standing", steps, true);
            } else {
                play_steps("Standing",
                           {{"Damp", sport_calls_.damp},
                            {"Squat2StandUp", [this] { return squat_to_stand(); }}},
                           true);
            }
        }
    }

    // The biped's FSM needs a moment after Damp before the stand takes.
    int32_t squat_to_stand() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return sport_calls_.squat2stand_up();
    }

    // Run the SDK calls in order and claim `status` only if every one returned 0.
    // /robot_status is a statement about the chassis, and a false one is worse than
    // none: a refusing sport service used to be reported as a successful stand.
    void play_steps(const std::string& what,
                    const std::vector<std::pair<std::string, std::function<int32_t()>>>& steps,
                    bool standing) {
        std::string said;
        bool all_ok = true;
        for (const auto& [name, call] : steps) {
            int32_t code = -1;
            std::string note;
            if (call) {
                code = call();
            } else {
                // The C++ LocoClient exposes FSM primitives only (no python-style
                // Squat2StandUp wrapper); an unbound call is reported, never guessed.
                note = " (unavailable in this SDK build)";
            }
            if (!said.empty()) {
                said += ", ";
            }
            said += name + " code=" + std::to_string(code) + note;
            all_ok = all_ok && code == 0;
        }
        if (all_ok) {
            RCLCPP_INFO(get_logger(), "%s: %s", what.c_str(), said.c_str());
            robot_status_ = standing;
        } else {
            RCLCPP_ERROR(get_logger(),
                         "%s REFUSED by the robot: %s. The commands reached the sport "
                         "service and it declined them.",
                         what.c_str(), said.c_str());
        }
    }

    // --- rt/lowstate: battery, throttled to 1 Hz ----------------------------
    void LowStateMessageHandler(const unitree_go::msg::dds_::LowState_& msg) {
        const double now = monotonic_s();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (last_battery_time_.has_value() &&
                now - *last_battery_time_ < kBatteryPeriodS) {
                return;
            }
            last_battery_time_ = now;
        }
        try {
            std_msgs::msg::Float32 battery_msg;
            battery_msg.data = static_cast<float>(msg.bms_state().soc());
            publisher_battery_->publish(battery_msg);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error in LowStateMessageHandler: %s", e.what());
        }
    }

    // --- rt/utlidar/robot_odom: chassis odometry, capped at 50 Hz ----------
    // Already a nav_msgs Odometry on the wire; the 50 Hz cap matches python.
    // Restamped on the ROS clock: the chassis stamp is its own timebase. The
    // chassis's own odom origin; unrelated to tinynav's "world".
    void ChassisOdomMessageHandler(const nav_msgs::msg::dds_::Odometry_& msg) {
        const double tnow = monotonic_s();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (tnow - last_chassis_odom_time_ < kChassisOdomPeriodS) {
                return;
            }
            last_chassis_odom_time_ = tnow;
        }
        try {
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = now();
            odom.header.frame_id = "odom";
            odom.child_frame_id = "base_link";
            odom.pose.pose.position.x = msg.pose().pose().position().x();
            odom.pose.pose.position.y = msg.pose().pose().position().y();
            odom.pose.pose.position.z = msg.pose().pose().position().z();
            odom.pose.pose.orientation.x = msg.pose().pose().orientation().x();
            odom.pose.pose.orientation.y = msg.pose().pose().orientation().y();
            odom.pose.pose.orientation.z = msg.pose().pose().orientation().z();
            odom.pose.pose.orientation.w = msg.pose().pose().orientation().w();
            odom.twist.twist.linear.x = msg.twist().twist().linear().x();
            odom.twist.twist.linear.y = msg.twist().twist().linear().y();
            odom.twist.twist.linear.z = msg.twist().twist().linear().z();
            odom.twist.twist.angular.z = msg.twist().twist().angular().z();
            publisher_chassis_odom_->publish(std::move(odom));
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error in ChassisOdomMessageHandler: %s", e.what());
        }
    }

    // --- 1 Hz cached status -------------------------------------------------
    void publish_robot_status() {
        std_msgs::msg::String msg;
        msg.data = robot_status_string(robot_status_);
        publisher_robot_status_->publish(msg);
    }

    // --- setup --------------------------------------------------------------
    void build_sport_client() {
        if (robot_model_ == "go2" || robot_model_ == "go2w") {
            holder_.go2 = std::make_unique<unitree::robot::go2::SportClient>();
            sport_client_ = holder_.go2.get();
            sport_calls_.classic_walk = [this] { return holder_.go2->ClassicWalk(true); };
            sport_calls_.stand_up = [this] { return holder_.go2->StandUp(); };
            sport_calls_.stand_down = [this] { return holder_.go2->StandDown(); };
            sport_calls_.balance_stand = [this] { return holder_.go2->BalanceStand(); };
            sport_calls_.move = [this](float vx, float vy, float wz) {
                return holder_.go2->Move(vx, vy, wz);
            };
        } else if (robot_model_ == "b2" || robot_model_ == "b2w") {
            holder_.b2 = std::make_unique<unitree::robot::b2::SportClient>();
            sport_client_ = holder_.b2.get();
            sport_calls_.classic_walk = [this] { return holder_.b2->ClassicWalk(true); };
            sport_calls_.stand_up = [this] { return holder_.b2->StandUp(); };
            sport_calls_.stand_down = [this] { return holder_.b2->StandDown(); };
            sport_calls_.balance_stand = [this] { return holder_.b2->BalanceStand(); };
            sport_calls_.switch_gait = [this] { return holder_.b2->SwitchGait(1); };
            sport_calls_.move = [this](float vx, float vy, float wz) {
                return holder_.b2->Move(vx, vy, wz);
            };
        } else {  // g1
            holder_.g1 = std::make_unique<unitree::robot::g1::LocoClient>();
            sport_client_ = holder_.g1.get();
            sport_calls_.damp = [this] { return holder_.g1->Damp(); };
            // The C++ LocoClient exposes FSM primitives (SetFsmId) but not the
            // python-style Squat2StandUp/StandUp2Squat wrappers; those stay
            // unbound and play_steps reports them as unavailable. g1 is not a
            // fleet model — wire real FSM ids before field use.
            sport_calls_.move = [this](float vx, float vy, float wz) {
                return holder_.g1->Move(vx, vy, wz);
            };
        }
    }

    // --- state --------------------------------------------------------------
    std::string network_interface_;
    std::string robot_model_;
    bool is_quadruped_ = true;
    bool has_switch_gait_ = false;

    SportClientHolder holder_;
    unitree::robot::Client* sport_client_ = nullptr;
    SportCalls sport_calls_;
    std::unique_ptr<GaitWorker> gait_worker_;

    std::unique_ptr<unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Twist_>> twist_subscriber_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>> action_subscriber_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>> lowstate_subscriber_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<nav_msgs::msg::dds_::Odometry_>> chassis_odom_subscriber_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>> sport_state_subscriber_;

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_battery_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_robot_status_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_chassis_odom_;
    rclcpp::TimerBase::SharedPtr watch_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;

    ChassisWatch watch_{get_logger()};
    // Shared between the SDK reader threads and the executor thread.
    std::mutex state_mutex_;
    bool robot_status_ = false;  // false = SITTING, true = STANDUP
    bool walking_ = false;
    bool gait_due_ = false;
    int32_t move_failures_ = 0;
    std::optional<double> last_twist_time_;
    std::optional<double> move_failure_logged_at_;
    std::optional<double> last_sport_state_time_;
    std::optional<double> last_battery_time_;
    double last_chassis_odom_time_ = 0.0;
};

int main(int argc, char** argv) {
    // Fleet CLI parity: --network-interface stays a plain argument (the dog's
    // run scripts call it that way); everything else goes to rclcpp.
    std::string interface_override;
    std::vector<std::string> ros_args;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--network-interface" && i + 1 < argc) {
            interface_override = argv[++i];
        } else {
            ros_args.push_back(argv[i]);
        }
    }
    // The constructor reads network_interface, so the override must be in
    // place before the node exists.
    rclcpp::NodeOptions options;
    if (!interface_override.empty()) {
        options.parameter_overrides({rclcpp::Parameter("network_interface", interface_override)});
    }
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UnitreeControlNode>(options);
    tinynav::logging_setup::install();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
