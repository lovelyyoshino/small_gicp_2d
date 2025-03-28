// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "traits.hpp"
#include "../traits_gicp.hpp"
#include "incremental_voxelmap.hpp"

namespace small_gicp {

/// @brief Gaussian voxel that computes and stores voxel mean and covariance.
struct GaussianVoxel {
public:
  struct Setting {};

  /// @brief Constructor.
  GaussianVoxel() : finalized(false), num_points(0), mean(Eigen::Vector3d::Zero()), cov(Eigen::Matrix3d::Zero()) {}

  /// @brief  Number of points in the voxel (Always 1 for GaussianVoxel).
  size_t size() const { return 1; }

  /// @brief  Add a point to the voxel.
  /// @param  setting        Setting
  /// @param  transformed_pt Transformed point mean
  /// @param  points         Point cloud
  /// @param  i              Index of the point
  /// @param  T              Transformation matrix
  template <typename PointCloud>
  void add(const Setting& setting, const Eigen::Vector3d& transformed_pt, const PointCloud& points, size_t i, const Eigen::Isometry2d& T) {
    if (finalized) {
      this->finalized = false;
      this->mean *= num_points;
      this->cov *= num_points;
    }

    num_points++;
    this->mean += transformed_pt;
    this->cov += T.matrix() * traits::cov(points, i) * T.matrix().transpose();
  }

  /// @brief Finalize the voxel mean and covariance.
  void finalize() {
    if (finalized) {
      return;
    }

    mean /= num_points;
    cov /= num_points;
    finalized = true;
  }

public:
  bool finalized;        ///< If true, mean and cov are finalized, otherwise they represent the sum of input points
  size_t num_points;     ///< Number of input points
  Eigen::Vector3d mean;  ///< Mean
  Eigen::Matrix3d cov;   ///< Covariance
};

namespace traits {

template <>
struct Traits<GaussianVoxel> {
  static size_t size(const GaussianVoxel& voxel) { return 1; }
  static bool has_points(const GaussianVoxel& voxel) { return true; }
  static bool has_covs(const GaussianVoxel& voxel) { return true; }

  static const Eigen::Vector3d& point(const GaussianVoxel& voxel, size_t i) { return voxel.mean; }
  static const Eigen::Matrix3d& cov(const GaussianVoxel& voxel, size_t i) { return voxel.cov; }

  static size_t nearest_neighbor_search(const GaussianVoxel& voxel, const Eigen::Vector3d& pt, size_t* k_index, double* k_sq_dist) {
    *k_index = 0;
    *k_sq_dist = (voxel.mean - pt).squaredNorm();
    return 1;
  }

  static size_t knn_search(const GaussianVoxel& voxel, const Eigen::Vector3d& pt, size_t k, size_t* k_index, double* k_sq_dist) {
    return nearest_neighbor_search(voxel, pt, k_index, k_sq_dist);
  }

  template <typename Result>
  static void knn_search(const GaussianVoxel& voxel, const Eigen::Vector3d& pt, Result& result) {
    result.push(0, (voxel.mean - pt).squaredNorm());
  }
};

}  // namespace traits

using GaussianVoxelMap = IncrementalVoxelMap<GaussianVoxel>;

}  // namespace small_gicp
