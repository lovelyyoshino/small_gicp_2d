// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace small_gicp {

/// @brief Create skew symmetric matrix
/// @param x  Rotation vector
/// @return   Skew symmetric matrix
inline Eigen::Matrix3d skew(const Eigen::Vector3d& x) {
  Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
  skew(0, 1) = -x[2];
  skew(0, 2) = x[1];
  skew(1, 0) = x[2];
  skew(1, 2) = -x[0];
  skew(2, 0) = -x[1];
  skew(2, 1) = x[0];

  return skew;
}

/*
 * SO3 expmap code taken from Sophus
 * https://github.com/strasdat/Sophus/blob/593db47500ea1a2de5f0e6579c86147991509c59/sophus/so3.hpp#L585
 *
 * Copyright 2011-2017 Hauke Strasdat
 *           2012-2017 Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// @brief SO3 expmap.
/// @param omega  [rx, ry, rz]
/// @return       Quaternion
inline Eigen::Quaterniond so3_exp(const Eigen::Vector3d& omega) {
  double theta_sq = omega.dot(omega);

  double imag_factor;
  double real_factor;
  if (theta_sq < 1e-10) {
    double theta_quad = theta_sq * theta_sq;
    imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
    real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
  } else {
    double theta = std::sqrt(theta_sq);
    double half_theta = 0.5 * theta;
    imag_factor = std::sin(half_theta) / theta;
    real_factor = std::cos(half_theta);
  }

  return Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
}

// Rotation-first
/// @brief SE3 expmap (Rotation-first).
/// @param a  Twist vector [rx, ry, rz, tx, ty, tz]
/// @return   SE3 matrix
inline Eigen::Isometry3d se3_exp(const Eigen::Matrix<double, 6, 1>& a) {
  const Eigen::Vector3d omega = a.head<3>();

  const double theta_sq = omega.dot(omega);
  const double theta = std::sqrt(theta_sq);

  Eigen::Isometry3d se3 = Eigen::Isometry3d::Identity();
  se3.linear() = so3_exp(omega).toRotationMatrix();

  if (theta < 1e-10) {
    se3.translation() = se3.linear() * a.tail<3>();
    /// Note: That is an accurate expansion!
  } else {
    const Eigen::Matrix3d Omega = skew(omega);
    const Eigen::Matrix3d V = (Eigen::Matrix3d::Identity() + (1.0 - std::cos(theta)) / theta_sq * Omega + (theta - std::sin(theta)) / (theta_sq * theta) * Omega * Omega);
    se3.translation() = V * a.tail<3>();
  }

  return se3;
}

// SO2 exponential map
/// @brief SO2 expmap.
/// @param theta  Rotation angle
/// @return       Quaternion (representing 2D rotation)
inline Eigen::Quaterniond so2_exp(double theta) {
    double real_part = std::cos(theta);
    double imag_part = std::sin(theta);

    // For SO2, we create a quaternion representing a rotation around the Z-axis.
    // Since SO2 is a 2D rotation, we use [w, 0, 0, z] form.
    return Eigen::Quaterniond(real_part, 0.0, 0.0, imag_part);
}


// SE2 exponential map
/// @brief SE2 expmap.
/// @param a  Twist vector [rz, tx, ty]
/// @return   SE2 transformation (Isometry2d)
#include <Eigen/Dense>
#include <cmath>

inline Eigen::Isometry2d se2_exp(const Eigen::Matrix<double, 3, 1>& v) {
    double theta = v[0]; // 使用 v[0] 作为旋转角度
    Eigen::Isometry2d se2 = Eigen::Isometry2d::Identity();

    // 使用 SO(2) 的指数映射计算旋转矩阵
    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << std::cos(theta), -std::sin(theta),
        std::sin(theta), std::cos(theta);
    se2.linear() = rotation_matrix;

    // 如果旋转角度接近零，使用泰勒展开近似
    if (std::abs(theta) < 1e-10) {
        // 使用泰勒展开的近似
        double theta_sq = theta * theta;
        double sin_theta_by_theta = 1.0 - (1.0 / 6.0) * theta_sq;
        double one_minus_cos_theta_by_theta = (0.5) * theta - (1.0 / 24.0) * theta * theta_sq;

        // 设置平移部分
        Eigen::Vector2d t;
        t.x() = sin_theta_by_theta * v[1] - one_minus_cos_theta_by_theta * v[2];
        t.y() = one_minus_cos_theta_by_theta * v[1] + sin_theta_by_theta * v[2];
        se2.translation() = t;
    } else {
        // 计算 a 和 b 用于修正平移部分
        double a = std::sin(theta) / theta;  // sin(theta) / theta
        double b = (1.0 - std::cos(theta)) / theta;  // (1 - cos(theta)) / theta

        // 计算并设置平移部分
        Eigen::Vector2d t;
        t.x() = a * v[1] - b * v[2];
        t.y() = b * v[1] + a * v[2];
        se2.translation() = t;
    }

    return se2;
}





}  // namespace small_gicp
