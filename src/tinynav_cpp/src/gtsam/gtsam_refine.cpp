// Port of reference/tinynav/core/perception_node.py's [ISAM Processing] block.
// Graph rebuilt from scratch every frame over the keyframe window (the python's
// "we have new graph each time"; NOT an iSAM window despite the log names):
//   - every keyframe: B(i) at ConstantBias() with a 1e-2 prior, V(i), X(i)
//   - first keyframe only: PriorFactorPose3 sigma 1e-1
//   - consecutive pairs: CombinedImuFactor on the batch-preintegrated window
//   - failed pair-PnP: PriorFactorVector(V(i), 0, sigma 0.25)
//   - one SmartStereoProjectionPoseFactor per visual track (isotropic 1.0 px)
//   - LevenbergMarquardt, max 3 iterations, result written back in place
#include "tinynav_cpp/gtsam/refine.hpp"

#include <cmath>
#include <cstdio>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

namespace tinynav::gtsam
{

namespace
{

using ::gtsam::Symbol;

// Matrix4x4ToGtsamPose3.
::gtsam::Pose3 pose_from_matrix(const Eigen::Matrix4d & T)
{
  return ::gtsam::Pose3(::gtsam::Rot3(T.topLeftCorner<3, 3>()), ::gtsam::Point3(T.topRightCorner<3, 1>()));
}

// Batch preintegration for one pair: samples with t_i < stamp <= t_{i+1}, each
// dt = stamp - previous stamp (previous = t_i for the first). See the
// RefineInput::imu note for the python-divergence rationale.
::gtsam::PreintegratedCombinedMeasurements preintegrate_pair(
  const std::shared_ptr<::gtsam::PreintegrationCombinedParams> & params,
  const std::vector<core::ImuSample> & imu, double t_i, double t_j)
{
  ::gtsam::PreintegratedCombinedMeasurements pim(params, ::gtsam::imuBias::ConstantBias());
  double prev_stamp = t_i;
  for (const core::ImuSample & s : imu) {
    if (s.stamp <= t_i) continue;
    if (s.stamp > t_j) break;
    // python's drain skips non-increasing stamps ("should only happen at
    // beginning") — a negative dt would poison the preintegration covariance.
    if (s.stamp <= prev_stamp) continue;
    pim.integrateMeasurement(s.accel, s.gyro, s.stamp - prev_stamp);
    prev_stamp = s.stamp;
  }
  return pim;
}

class GtsamRefineImpl : public Refine
{
  public:
    bool available() const override { return true; }

    Metrics refine(std::vector<Eigen::Matrix4d> & poses,
                   std::vector<Eigen::Vector3d> & velocities,
                   const RefineInput & input) override
    {
      Metrics metrics;
      const int n = static_cast<int>(poses.size());
      metrics.num_keyframes = n;
      metrics.num_tracks = static_cast<int>(input.tracks.size());
      if (n == 0 || static_cast<int>(input.keyframe_timestamps.size()) != n) {
        return metrics;
      }

      // python: gtsam.PreintegrationCombinedParams.MakeSharedU() (Z-up, g=9.81)
      const auto params = ::gtsam::PreintegrationCombinedParams::MakeSharedU();

      ::gtsam::NonlinearFactorGraph graph;
      ::gtsam::Values initial;

      const ::gtsam::imuBias::ConstantBias zero_bias;
      for (int i = 0; i < n; ++i) {
        initial.insert(Symbol('b', i), zero_bias);
        graph.emplace_shared<::gtsam::PriorFactor<::gtsam::imuBias::ConstantBias>>(
          Symbol('b', i), zero_bias,
          ::gtsam::noiseModel::Diagonal::Sigmas(
            (::gtsam::Vector6() << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()));
        initial.insert(Symbol('v', i),
                      ::gtsam::Vector(velocities[static_cast<size_t>(i)]));
        initial.insert(Symbol('x', i), pose_from_matrix(poses[static_cast<size_t>(i)]));
        if (i == 0) {
          graph.emplace_shared<::gtsam::PriorFactor<::gtsam::Pose3>>(
            Symbol('x', 0), pose_from_matrix(poses[0]),
            ::gtsam::noiseModel::Diagonal::Sigmas(
              (::gtsam::Vector6() << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()));
        }
        if (i != n - 1) {
          graph.emplace_shared<::gtsam::CombinedImuFactor>(
            Symbol('x', i), Symbol('v', i), Symbol('x', i + 1), Symbol('v', i + 1),
            Symbol('b', i), Symbol('b', i + 1),
            preintegrate_pair(params, input.imu, input.keyframe_timestamps[static_cast<size_t>(i)],
                              input.keyframe_timestamps[static_cast<size_t>(i) + 1]));
        }
      }

      for (int i : input.velocity_prior_pose_idx) {
        // python PriorFactorVector = PriorFactor<Vector> (dynamic dim)
        graph.emplace_shared<::gtsam::PriorFactor<::gtsam::Vector>>(
          Symbol('v', i), ::gtsam::Vector3::Zero(),
          ::gtsam::noiseModel::Diagonal::Sigmas(
            (::gtsam::Vector3() << 0.25, 0.25, 0.25).finished()));
      }

      // One smart factor per track; python: Isotropic Sigma(3, 1.0) +
      // default SmartProjectionParams, calib per observation (same value).
      const auto calib = std::make_shared<::gtsam::Cal3_S2Stereo>(
        input.K(0, 0), input.K(1, 1), 0.0, input.K(0, 2), input.K(1, 2), input.baseline);
      for (const std::vector<SmartObservation> & track : input.tracks) {
        if (track.size() < 2) continue;
        auto factor = std::make_shared<::gtsam::SmartStereoProjectionPoseFactor>(
          ::gtsam::noiseModel::Isotropic::Sigma(3, 1.0), ::gtsam::SmartProjectionParams());
        for (const SmartObservation & obs : track) {
          factor->add(::gtsam::StereoPoint2(obs.uL, obs.uR, obs.v),
                      Symbol('x', obs.pose_idx), calib);
        }
        graph.add(factor);
      }

      metrics.num_factors = static_cast<int>(graph.size());
      metrics.num_variables = static_cast<int>(initial.size());

      ::gtsam::LevenbergMarquardtParams lm_params;
      lm_params.setMaxIterations(3);  // python: params.setMaxIterations(3)
      ::gtsam::LevenbergMarquardtOptimizer lm(graph, initial, lm_params);
      const ::gtsam::Values result = lm.optimize();

      metrics.initial_error = graph.error(initial);
      if (!std::isfinite(metrics.initial_error)) {
        int bad = 0;
        for (size_t fi = 0; fi < graph.size() && bad < 5; ++fi) {
          const double e = graph[fi]->error(initial);
          if (!std::isfinite(e)) {
            const auto smart =
              dynamic_cast<const ::gtsam::SmartStereoProjectionPoseFactor*>(
                graph[fi].get());
            fprintf(stderr,
                    "[refine-dbg] factor %zu type=%s keys=%s error=%g\n", fi,
                    smart ? "smart" : "other",
                    smart ? "n/a" : "?", e);
            ++bad;
          }
        }
        fflush(stderr);
      }
      metrics.final_error = graph.error(result);
      for (int i = 0; i < n; ++i) {
        poses[static_cast<size_t>(i)] =
          result.at<::gtsam::Pose3>(::gtsam::Symbol('x', i)).matrix();
        velocities[static_cast<size_t>(i)] =
          result.at<::gtsam::Vector>(::gtsam::Symbol('v', i));
      }
      return metrics;
    }
};

}  // namespace

std::shared_ptr<Refine> make_gtsam_refine()
{
  return std::make_shared<GtsamRefineImpl>();
}

}  // namespace tinynav::gtsam
