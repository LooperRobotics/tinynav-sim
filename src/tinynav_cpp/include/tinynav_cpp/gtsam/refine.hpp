// Port of reference/tinynav/core/perception_node.py's [ISAM Processing] block —
// the per-frame window factor graph (bias priors, first-pose prior,
// CombinedImuFactor between window keyframes, SmartStereoProjectionPoseFactor
// per visual track, LM with 3 iterations). The interface keeps gtsam types out
// of the component; make_gtsam_refine() returns nullptr when the build has no
// GTSAM, and the component then degrades to the chained-PnP v1 behaviour.
#pragma once

#include <Eigen/Dense>

#include <memory>
#include <vector>

#include "tinynav_cpp/core/imu.hpp"

namespace tinynav::gtsam
{

// One smart-factor observation: the StereoPoint2 (uL, uL - disparity[v,u], v)
// seen by window keyframe pose_idx. The component builds these out of its
// stored per-keyframe SuperPoint results and disparity maps.
struct SmartObservation
{
  int pose_idx = 0;
  double uL = 0.0;
  double uR = 0.0;
  double v = 0.0;
};

// Everything the graph needs besides the poses/velocities it refines in place.
struct RefineInput
{
  std::vector<double> keyframe_timestamps;  // window order, seconds
  // IMU samples overlapping the window (stamp-ascending). The impl slices
  // (t_i, t_{i+1}] per pair and preintegrates with dt = stamp - prev_stamp
  // (prev = previous sample, or t_i for the pair's first). KNOWN DIVERGENCE
  // from the python: its per-frame drain re-integrates the peeked head sample
  // each stereo frame, double-counting about one sample interval per frame
  // (<10 ms at gz's IMU rate); this batch integrates every sample exactly once.
  std::vector<core::ImuSample> imu;
  // One entry per visual track (the UF sets with >= 2 observations).
  std::vector<std::vector<SmartObservation>> tracks;
  // Window indices i whose pair-PnP filter failed: python adds
  // PriorFactorVector(V(i), 0, sigma 0.25) for them.
  std::vector<int> velocity_prior_pose_idx;
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  double baseline = 0.0;
};

class Refine
{
  public:
    virtual ~Refine() = default;
    virtual bool available() const { return false; }

    struct Metrics
    {
        int num_keyframes = 0;
        int num_tracks = 0;
        int num_factors = 0;
        int num_variables = 0;
        double initial_error = 0.0;
        double final_error = 0.0;
    };

    // poses/velocities are the window in order; refine mutates them in place
    // (the python writes result.atPose3(X(i)) back into keyframe.pose/.velocity).
    virtual Metrics refine(std::vector<Eigen::Matrix4d> & poses,
                           std::vector<Eigen::Vector3d> & velocities,
                           const RefineInput & input) = 0;
};

// nullptr when compiled without GTSAM.
std::shared_ptr<Refine> make_gtsam_refine();

}  // namespace tinynav::gtsam
