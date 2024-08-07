// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace small_gicp {

/// @brief Registration result
struct RegistrationResult {
  RegistrationResult(const Eigen::Isometry2d& T = Eigen::Isometry2d::Identity())
  : T_target_source(T),
    converged(false),
    iterations(0),
    num_inliers(0),
    H(Eigen::Matrix<double, 3, 3>::Zero()),
    b(Eigen::Matrix<double, 3, 1>::Zero()),
    error(0.0) {}

  Eigen::Isometry2d T_target_source;  ///<  Estimated transformation

  bool converged;      ///< If the optimization converged
  size_t iterations;   ///< Number of optimization iterations
  size_t num_inliers;  ///< Number of inliear points

  Eigen::Matrix<double, 3, 3> H;  ///< Final information matrix
  Eigen::Matrix<double, 3, 1> b;  ///< Final information vector
  double error;                   ///< Final error
};

}  // namespace small_gicp
