// Port of reference/tinynav/core/imu_propagator_node.py::integrate
#include "tinynav_cpp/core/imu.hpp"

#include "tinynav_cpp/core/math.hpp"

namespace tinynav::core {

OdomState integrate(const OdomState& odom_prev, const ImuSample& imu,
                    const Eigen::Vector3d& gravity_world) {
    const Eigen::Matrix3d rotation_prev = odom_prev.pose.topLeftCorner<3, 3>();
    const Eigen::Vector3d translation_prev = odom_prev.pose.topRightCorner<3, 1>();

    const double dt = imu.stamp - odom_prev.stamp;
    // Python: scipy R.from_rotvec(gyro * dt).as_matrix(), which goes
    // rotvec -> quat -> matrix; rotvec_to_matrix is the same Rodrigues
    // rotation and agrees with it to machine epsilon.
    const Eigen::Matrix3d delta_rotation = rotvec_to_matrix(imu.gyro * dt);
    const Eigen::Matrix3d rotation_new = rotation_prev * delta_rotation;
    const Eigen::Vector3d accel_world = rotation_prev * imu.accel + gravity_world;
    const Eigen::Vector3d velocity_new = odom_prev.velocity + accel_world * dt;
    const Eigen::Vector3d translation_new =
        translation_prev + odom_prev.velocity * dt + 0.5 * accel_world * dt * dt;

    OdomState odom_new;
    odom_new.stamp = imu.stamp;
    odom_new.pose.setIdentity();
    odom_new.pose.topLeftCorner<3, 3>() = rotation_new;
    odom_new.pose.topRightCorner<3, 1>() = translation_new;
    odom_new.velocity = velocity_new;
    odom_new.gyro = imu.gyro;
    return odom_new;
}

}  // namespace tinynav::core
