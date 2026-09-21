// Port of reference/tinynav/core/imu_propagator_node.py::integrate — math only.
// The rclpy node shell (buffers, subscriptions, publishing) is not ported here;
// it belongs to the component layer. Timestamps are plain seconds, poses and
// velocities are Eigen types instead of ROS messages.
#pragma once

#include <Eigen/Dense>

namespace tinynav::core {

// One IMU reading. stamp in seconds (Python: header.stamp as sec + nsec*1e-9).
struct ImuSample {
    double stamp = 0.0;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // angular velocity, rad/s
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // linear acceleration, m/s^2
};

// Odometry state carried between integrate() steps (the pure-data form of the
// Odometry msg the Python unpacks via msg2np and rebuilds via np2msg).
struct OdomState {
    double stamp = 0.0;                                         // seconds
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();         // body-from-world T
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();         // linear, world frame
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();             // last IMU gyro sample
};

// Port of reference/tinynav/core/imu_propagator_node.py::integrate
// One Euler pre-integration step: rotation by rotvec(gyro*dt), acceleration
// rotated into the world frame plus gravity, trapezoidal-free position update
// identical to the Python. The output stamps itself with imu.stamp and
// carries the raw gyro through, as np2msg does.
OdomState integrate(const OdomState& odom_prev, const ImuSample& imu,
                    const Eigen::Vector3d& gravity_world = Eigen::Vector3d(0.0, 0.0, -9.80));

}  // namespace tinynav::core
