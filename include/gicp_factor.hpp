// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ann/traits.hpp"
#include "traits_gicp.hpp"
#include "lie.hpp"

namespace small_gicp {

/// @brief GICP (distribution-to-distribution) per-point error factor.
struct GICPFactor {
  struct Setting {};

  /// @brief Constructor
  GICPFactor(const Setting& setting = Setting())
  : target_index(std::numeric_limits<size_t>::max()),
    source_index(std::numeric_limits<size_t>::max()),
    mahalanobis(Eigen::Matrix2d::Zero()) {}

  // 计算旋转矩阵
  Eigen::Matrix2d rotationMatrix(double theta) {
      Eigen::Matrix2d R;
      R << std::cos(theta), -std::sin(theta),
          std::sin(theta), std::cos(theta);
      return R;
  }

  /// @brief Linearize the factor
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param T            Linearization point
  /// @param source_index Source point index
  /// @param rejector     Correspondence rejector
  /// @param H            Linearized information matrix
  /// @param b            Linearized information vector
  /// @param e            Error at the linearization point
  /// @return             True if the point is inlier
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
  bool linearize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry2d& T,
    size_t source_index,
    const CorrespondenceRejector& rejector,
    Eigen::Matrix<double, 3, 3>* H,
    Eigen::Matrix<double, 3, 1>* b,
    double* e) {
    //
    this->source_index = source_index;
    this->target_index = std::numeric_limits<size_t>::max();

    const Eigen::Vector3d transed_source_pt = T * traits::point(source, source_index);

    size_t k_index;
    double k_sq_dist;
    if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &k_index, &k_sq_dist) || rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false;
    }

    target_index = k_index;
    const Eigen::Matrix3d RCR = traits::cov(target, target_index) + T.matrix() * traits::cov(source, source_index) * T.matrix().transpose();
    mahalanobis.block<2, 2>(0, 0) = RCR.block<2, 2>(0, 0).inverse();

    const Eigen::Vector2d residual = (traits::point(target, target_index) - transed_source_pt).template head<2>();
        //    Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
//    J.block<3, 3>(0, 0) = T.linear() * skew(traits::point(source, source_index).template head<3>());
//    J.block<3, 3>(0, 3) = -T.linear();
//
//    *H = J.transpose() * mahalanobis * J;
//    *b = J.transpose() * mahalanobis * residual;
//    *e = 0.5 * residual.transpose() * mahalanobis * residual;


    // 假设 J 是一个适当大小的 Eigen 矩阵
    Eigen::Matrix<double, 2, 3> J= Eigen::Matrix<double, 2, 3>::Zero(); // 例子：2行3列的矩阵

    // hat{1} = 0 ,-1
    //          1 , 0;
    Eigen::Matrix2d R_hat_1;
    R_hat_1 << 0, -1,
              1, 0;

    // 提取点
    Eigen::Vector2d point = traits::point(source, source_index).template head<2>(); // 提取点

    // 更新 J 的内容: https://arxiv.org/pdf/1812.01537 (129)
    J.block<2, 1>(0, 0) = -T.linear() * R_hat_1 * point;
    J.block<2, 2>(0, 1) = -T.linear();


    *H = J.transpose() * mahalanobis * J;
    *b = J.transpose() * mahalanobis * residual;
    *e = 0.5 * residual.transpose() * mahalanobis * residual;

    return true;
  }

  /// @brief Evaluate error
  /// @param target   Target point cloud
  /// @param source   Source point cloud
  /// @param T        Evaluation point
  /// @return Error
  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry2d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    const Eigen::Vector3d transed_source_pt = T * traits::point(source, source_index);
    const Eigen::Vector2d residual = (traits::point(target, target_index) - transed_source_pt).template head<2>();
    return 0.5 * residual.transpose() * mahalanobis * residual;
  }

  /// @brief Returns true if this factor is not rejected as an outlier
  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }

  size_t target_index;          ///< Target point index
  size_t source_index;          ///< Source point index
  Eigen::Matrix2d mahalanobis;  ///< Fused precision matrix
};
}  // namespace small_gicp
