// Port of reference/tinynav/core/imu_propagator_node.py::ImuPropagatorNode —
// the rclcpp component shell; the math is core/imu.hpp::integrate.
// Buffers, callback flow and the 50 ms publish gate follow the Python exactly.
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "tinynav_cpp/core/imu.hpp"

namespace tinynav {

class ImuPropagatorComponent : public rclcpp::Node {
  public:
    explicit ImuPropagatorComponent(const rclcpp::NodeOptions& options)
        : Node("imu_propagator_node", options) {
        rclcpp::QoS imu_qos(rclcpp::KeepLast(1000));
        imu_qos.best_effort();
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/camera/camera/imu", imu_qos,
            [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { imu_callback(*msg); });
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/slam/odometry_visual", imu_qos,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { odom_callback(*msg); });
        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", 50);
    }

  private:
    static double stamp_to_sec(const builtin_interfaces::msg::Time& stamp) {
        return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
    }

    // Port of ImuPropagatorNode.imu_callback.
    void imu_callback(const sensor_msgs::msg::Imu& imu_msg) {
        if (odom_100hz_buffer_.empty()) {
            return;
        }

        const double timestamp = stamp_to_sec(imu_msg.header.stamp);
        imu_buffer_.emplace_back(timestamp, imu_msg);
        if (imu_buffer_.size() > 2000) {
            imu_buffer_.pop_front();
        }

        if (imu_buffer_.back().first <= odom_100hz_buffer_.back().first + 0.050) {
            return;
        }

        // Newest-first scan for the first IMU sample newer than the last odom.
        long start_idx = -1;  // Python None
        for (size_t i = 0; i < imu_buffer_.size(); ++i) {
            if (imu_buffer_[imu_buffer_.size() - (i + 1)].first >
                odom_100hz_buffer_.back().first) {
                start_idx = static_cast<long>(imu_buffer_.size() - (i + 1));
            } else {
                break;
            }
        }
        if (start_idx < 0) {
            return;
        }

        for (size_t i = static_cast<size_t>(start_idx); i < imu_buffer_.size(); ++i) {
            core::OdomState prev = to_state(odom_100hz_buffer_.back());
            core::ImuSample sample = to_sample(imu_buffer_[i].second);
            odom_100hz_buffer_.emplace_back(
                imu_buffer_[i].first,
                from_state(core::integrate(prev, sample), imu_buffer_[i].second.header.stamp,
                           odom_100hz_buffer_.back().second));
            if (odom_100hz_buffer_.size() > 1000) {
                odom_100hz_buffer_.pop_front();
            }
        }

        odom_pub_->publish(
            std::make_unique<nav_msgs::msg::Odometry>(odom_100hz_buffer_.back().second));
    }

    // Port of ImuPropagatorNode.odom_callback.
    void odom_callback(const nav_msgs::msg::Odometry& msg) {
        const double timestamp = stamp_to_sec(msg.header.stamp);
        odom_10hz_buffer_.emplace_back(timestamp, msg);
        if (odom_10hz_buffer_.size() > 100) {
            odom_10hz_buffer_.pop_front();
        }

        while (!odom_100hz_buffer_.empty() &&
               odom_100hz_buffer_.back().first > timestamp) {
            odom_100hz_buffer_.pop_back();
        }
        odom_100hz_buffer_.emplace_back(timestamp, msg);
    }

    static core::OdomState to_state(const std::pair<double, nav_msgs::msg::Odometry>& entry) {
        core::OdomState state;
        state.stamp = entry.first;
        const auto& pose = entry.second.pose.pose;
        state.pose.setIdentity();
        state.pose(0, 3) = pose.position.x;
        state.pose(1, 3) = pose.position.y;
        state.pose(2, 3) = pose.position.z;
        // msg2np: scipy R.from_quat (normalises) -> matrix.
        const Eigen::Quaterniond quat(pose.orientation.w, pose.orientation.x,
                                      pose.orientation.y, pose.orientation.z);
        state.pose.topLeftCorner<3, 3>() =
            quat.normalized().toRotationMatrix();
        state.velocity << entry.second.twist.twist.linear.x,
            entry.second.twist.twist.linear.y, entry.second.twist.twist.linear.z;
        return state;
    }

    static core::ImuSample to_sample(const sensor_msgs::msg::Imu& msg) {
        core::ImuSample sample;
        sample.stamp = stamp_to_sec(msg.header.stamp);
        sample.gyro << msg.angular_velocity.x, msg.angular_velocity.y,
            msg.angular_velocity.z;
        sample.accel << msg.linear_acceleration.x, msg.linear_acceleration.y,
            msg.linear_acceleration.z;
        return sample;
    }

    // np2msg + the raw gyro pass-through (integrate returns the same fields).
    static nav_msgs::msg::Odometry from_state(
        const core::OdomState& state, const builtin_interfaces::msg::Time& stamp,
        const nav_msgs::msg::Odometry& prev_msg) {
        nav_msgs::msg::Odometry out;
        out.header.stamp = stamp;
        out.header.frame_id = prev_msg.header.frame_id;
        out.child_frame_id = prev_msg.child_frame_id;
        const Eigen::Quaterniond quat(
            Eigen::Matrix3d(state.pose.topLeftCorner<3, 3>()));
        out.pose.pose.position.x = state.pose(0, 3);
        out.pose.pose.position.y = state.pose(1, 3);
        out.pose.pose.position.z = state.pose(2, 3);
        out.pose.pose.orientation.x = quat.x();
        out.pose.pose.orientation.y = quat.y();
        out.pose.pose.orientation.z = quat.z();
        out.pose.pose.orientation.w = quat.w();
        out.twist.twist.linear.x = state.velocity[0];
        out.twist.twist.linear.y = state.velocity[1];
        out.twist.twist.linear.z = state.velocity[2];
        out.twist.twist.angular.x = state.gyro[0];
        out.twist.twist.angular.y = state.gyro[1];
        out.twist.twist.angular.z = state.gyro[2];
        return out;
    }

    // (timestamp_sec, msg) pairs; std::deque gives both ends O(1) like the
    // Python list's append/pop(0) usage here.
    std::deque<std::pair<double, sensor_msgs::msg::Imu>> imu_buffer_;
    std::deque<std::pair<double, nav_msgs::msg::Odometry>> odom_10hz_buffer_;
    std::deque<std::pair<double, nav_msgs::msg::Odometry>> odom_100hz_buffer_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
};

// main.cpp's factory contract.
std::shared_ptr<rclcpp::Node> make_imu_propagator(
    const rclcpp::NodeOptions& options) {
    return std::make_shared<ImuPropagatorComponent>(options);
}

}  // namespace tinynav
