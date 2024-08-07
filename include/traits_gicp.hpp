
// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "traits_gicp.hpp"

namespace small_gicp {

/**
 * @brief Point cloud
 */
struct PointCloud {
public:
    using Ptr = std::shared_ptr<PointCloud>;
    using ConstPtr = std::shared_ptr<const PointCloud>;

    /// @brief Constructor
    PointCloud() {}

    /// @brief Constructor
    /// @param points  Points to initialize the point cloud
    template <typename T, int D, typename Allocator>
    explicit PointCloud(const std::vector<Eigen::Matrix<T, D, 1>, Allocator>& points) {
        this->resize(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            this->point(i) << points[i].template cast<double>().template head<2>(), 1.0;
        }
    }

    /// @brief Destructor
    ~PointCloud() {}

    /// @brief Number of points.
    size_t size() const { return points.size(); }

    /// @brief Check if the point cloud is empty.
    bool empty() const { return points.empty(); }

    /// @brief Resize point/normal/cov buffers.
    /// @param n  Number of points
    void resize(size_t n) {
        points.resize(n);
        normals.resize(n);
        covs.resize(n);
    }

    /// @brief Get i-th point.
    Eigen::Vector3d& point(size_t i) { return points[i]; }

    /// @brief Get i-th normal.
    Eigen::Vector3d& normal(size_t i) { return normals[i]; }

    /// @brief Get i-th covariance.
    Eigen::Matrix3d& cov(size_t i) { return covs[i]; }

    /// @brief Get i-th point (const).
    const Eigen::Vector3d& point(size_t i) const { return points[i]; }

    /// @brief Get i-th normal (const).
    const Eigen::Vector3d& normal(size_t i) const { return normals[i]; }

    /// @brief Get i-th covariance (const).
    const Eigen::Matrix3d& cov(size_t i) const { return covs[i]; }

public:
    std::vector<Eigen::Vector3d> points;   ///< Point coordinates (x, y, z, 1)
    std::vector<Eigen::Vector3d> normals;  ///< Point normals (nx, ny, nz, 0)
    std::vector<Eigen::Matrix3d> covs;     ///< Point covariances (3x3 matrix) + zero padding
};

namespace traits {

template <>
struct Traits<PointCloud> {
    using Points = PointCloud;

    static size_t size(const Points& points) { return points.size(); }

    static bool has_points(const Points& points) { return !points.points.empty(); }
    static bool has_normals(const Points& points) { return !points.normals.empty(); }
    static bool has_covs(const Points& points) { return !points.covs.empty(); }

    static const Eigen::Vector3d& point(const Points& points, size_t i) { return points.point(i); }
    static const Eigen::Vector3d& normal(const Points& points, size_t i) { return points.normal(i); }
    static const Eigen::Matrix3d& cov(const Points& points, size_t i) { return points.cov(i); }

    static void resize(Points& points, size_t n) { points.resize(n); }
    static void set_point(Points& points, size_t i, const Eigen::Vector3d& pt) { points.point(i) = pt; }
    static void set_normal(Points& points, size_t i, const Eigen::Vector3d& n) { points.normal(i) = n; }
    static void set_cov(Points& points, size_t i, const Eigen::Matrix3d& cov) { points.cov(i) = cov; }
};

}  // namespace traits

}  // namespace small_gicp
