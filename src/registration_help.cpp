// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include "../include/registration/registration_helper.hpp"

#include "../include/ann/kdtree.hpp"
#include "../include/ann/gaussian_voxelmap.hpp"
#include "../include/downsampling.hpp"
#include "../include/normal_estimation.hpp"
#include "../include/gicp_factor.hpp"
#include "../include/registration/reduction.hpp"
#include "../include/registration/registration.hpp"
#include "opencv2/opencv.hpp"

namespace small_gicp {

// Preprocess points
std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const PointCloud& points, double downsampling_resolution, int num_neighbors, int num_threads) {
    auto downsampled = voxelgrid_sampling(points, downsampling_resolution);
    auto kdtree = std::make_shared<KdTree<PointCloud>>(downsampled);
    estimate_normals_covariances(*downsampled, *kdtree, num_neighbors);
    return {downsampled, kdtree};
}

// Preprocess points with Eigen input
template <typename T, int D>
std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>>
preprocess_points(const std::vector<Eigen::Matrix<T, D, 1>>& points, double downsampling_resolution, int num_neighbors, int num_threads) {
    return preprocess_points(*std::make_shared<PointCloud>(points), downsampling_resolution, num_neighbors, num_threads);
}

// Explicit instantiation
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<float, 3, 1>>&, double, int, int);
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<float, 2, 1>>&, double, int, int);
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<double, 3, 1>>&, double, int, int);
template std::pair<PointCloud::Ptr, std::shared_ptr<KdTree<PointCloud>>> preprocess_points(const std::vector<Eigen::Matrix<double, 2, 1>>&, double, int, int);

// Create Gaussian voxel map
GaussianVoxelMap::Ptr create_gaussian_voxelmap(const PointCloud& points, double voxel_resolution) {
    auto voxelmap = std::make_shared<GaussianVoxelMap>(voxel_resolution);
    voxelmap->insert(points);
    return voxelmap;
}

// Align point clouds with Eigen input
template <typename T, int D>
RegistrationResult
align(const std::vector<Eigen::Matrix<T, D, 1>>& target, const std::vector<Eigen::Matrix<T, D, 1>>& source, const Eigen::Isometry2d& init_T, const RegistrationSetting& setting) {


    auto [target_points, target_tree] = preprocess_points(*std::make_shared<PointCloud>(target), setting.downsampling_resolution, 10, setting.num_threads);
    auto [source_points, source_tree] = preprocess_points(*std::make_shared<PointCloud>(source), setting.downsampling_resolution, 10, setting.num_threads);
//    cv::Mat target_img = cv::Mat::zeros(2000, 2000, CV_8UC3);
//    PointCloud target_point_all = *target_points;
//    for (int i =0; i<target_point_all.size(); i++){
//        if (target_point_all.point(i)[0]*10+1000<0 || target_point_all.point(i)[0]*10+1000>2000 || target_point_all.point(i)[1]*10+1000<0 || target_point_all.point(i)[1]*10+1000>2000){
//            continue;
//        }
//        cv::circle(target_img, cv::Point(target_point_all.point(i)[0]*10+1000, 1000-target_point_all.point(i)[1]*10), 1, cv::Scalar(255, 255, 255), -1);
//    }
//    PointCloud source_point_all = *source_points;
//    for (int i =0; i<source_point_all.size(); i++){
//        Eigen::Vector2d source_point_2d = init_T * Eigen::Vector2d(source_point_all.point(i)[0], source_point_all.point(i)[1]);
//        if(source_point_2d[0]*10+1000<0 || source_point_2d[0]*10+1000>2000 || source_point_2d[1]*10+1000<0 || source_point_2d[1]*10+1000>2000){
//            continue;
//        }
//        cv::circle(target_img, cv::Point(source_point_2d[0]*10+1000, 1000-source_point_2d[1]*10), 1, cv::Scalar(0, 0, 255), -1);
//    }
//    cv::imwrite("/home/ecarx/slam/target.png", target_img);
    if (setting.type == RegistrationSetting::VGICP) {
        auto target_voxelmap = create_gaussian_voxelmap(*target_points, setting.voxel_resolution);
        return align(*target_voxelmap, *source_points, init_T, setting);
    } else {
        return align(*target_points, *source_points, *target_tree, init_T, setting);
    }
}

template RegistrationResult
align(const std::vector<Eigen::Matrix<float, 3, 1>>&, const std::vector<Eigen::Matrix<float, 3, 1>>&, const Eigen::Isometry2d&, const RegistrationSetting&);
template RegistrationResult
align(const std::vector<Eigen::Matrix<float, 2, 1>>&, const std::vector<Eigen::Matrix<float, 2, 1>>&, const Eigen::Isometry2d&, const RegistrationSetting&);
template RegistrationResult
align(const std::vector<Eigen::Matrix<double, 3, 1>>&, const std::vector<Eigen::Matrix<double, 3, 1>>&, const Eigen::Isometry2d&, const RegistrationSetting&);
template RegistrationResult
align(const std::vector<Eigen::Matrix<double, 2, 1>>&, const std::vector<Eigen::Matrix<double, 2, 1>>&, const Eigen::Isometry2d&, const RegistrationSetting&);

// Align point clouds
RegistrationResult
align(const PointCloud& target, const PointCloud& source, const KdTree<PointCloud>& target_tree, const Eigen::Isometry2d& init_T, const RegistrationSetting& setting) {
    switch (setting.type) {
    default:
        std::cerr << "invalid registration type" << std::endl;
        abort();
    case RegistrationSetting::GICP: {
        Registration<GICPFactor, SerialReduction> registration;
//        registration.reduction.num_threads = setting.num_threads;
        registration.rejector.max_dist_sq = setting.max_correspondence_distance * setting.max_correspondence_distance;
        registration.criteria.rotation_eps = setting.rotation_eps;
        registration.criteria.translation_eps = setting.translation_eps;
        registration.optimizer.max_iterations = setting.max_iterations;
        registration.optimizer.verbose = setting.verbose;
        return registration.align(target, source, target_tree, init_T);
    }

    }
}

// Align point clouds with VGICP
RegistrationResult align(const GaussianVoxelMap& target, const PointCloud& source, const Eigen::Isometry2d& init_T, const RegistrationSetting& setting) {
    if (setting.type != RegistrationSetting::VGICP) {
        std::cerr << "invalid registration type for GaussianVoxelMap" << std::endl;
    }

    Registration<GICPFactor, SerialReduction> registration;
//    registration.reduction.num_threads = setting.num_threads;
    registration.criteria.rotation_eps = setting.rotation_eps;
    registration.criteria.translation_eps = setting.translation_eps;
    registration.optimizer.max_iterations = setting.max_iterations;
    registration.optimizer.verbose = setting.verbose;
    return registration.align(target, source, target, init_T);
}

}  // namespace small_gicp
