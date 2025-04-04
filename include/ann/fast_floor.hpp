// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace small_gicp {

/// @brief Fast floor (https://stackoverflow.com/questions/824118/why-is-floor-so-slow).
/// @param pt  Double vector
/// @return    Floored int vector
inline Eigen::Array3i fast_floor(const Eigen::Array3d& pt) {
  const Eigen::Array3i ncoord = pt.cast<int>();
  return ncoord - (pt < ncoord.cast<double>()).cast<int>();
};

}  // namespace small_gicp
