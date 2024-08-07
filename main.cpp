#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "registration/registration_helper.hpp"

int main() {
    // 初始化变换矩阵
    Eigen::Isometry2d init_T_target_source_tmp = Eigen::Isometry2d::Identity();
    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << 0.0668042, -0.997766, 
                       0.997766, 0.0668042;
    init_T_target_source_tmp.linear() = rotation_matrix;
    init_T_target_source_tmp.translation() = Eigen::Vector2d{-5.28473, 1.47331};
    
    std::cout << "--- init_T_target_source1 ---" << std::endl 
              << init_T_target_source_tmp.matrix() << std::endl;

    // 定义点云
    std::vector<Eigen::Vector2d> percep_pts_tmp;
    std::vector<Eigen::Vector2d> pPtsMap_tmp;

    // 添加地图点
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{15.12598, -4.9365783});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{5.842516, 4.397528});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-2.4041843, -1.674002});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-4.799726, -1.7386085});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-1.293435, 4.033495});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-10.053565, 3.92025});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-12.325812, 3.9082286});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-5.9963756, 3.948926});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-3.6760776, 3.963433});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{16.936523, -6.625896});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{3.3860104, 4.262526});
    pPtsMap_tmp.emplace_back(Eigen::Vector2d{-7.15835, -1.739128});

    // 添加感知点
    percep_pts_tmp.emplace_back(Eigen::Vector2d{3.1625, -2.0825});
    percep_pts_tmp.emplace_back(Eigen::Vector2d{-2.5874999, -3.645});
    percep_pts_tmp.emplace_back(Eigen::Vector2d{2.7000003, 4.405});
    percep_pts_tmp.emplace_back(Eigen::Vector2d{2.625, 6.755});
    percep_pts_tmp.emplace_back(Eigen::Vector2d{2.9625, 0.34249997});
    percep_pts_tmp.emplace_back(Eigen::Vector2d{-2.675, -1.1950002});
    percep_pts_tmp.emplace_back(Eigen::Vector2d{-2.7875, 1.2425001});

    // 执行配准
    small_gicp::RegistrationResult result = small_gicp::align(pPtsMap_tmp, percep_pts_tmp, init_T_target_source_tmp);

    // 输出结果
    std::cout << "iterations: " << result.iterations << std::endl;
    std::cout << "source: " << percep_pts_tmp.size() << std::endl;
    std::cout << "num_inliers: " << result.num_inliers << std::endl;
    std::cout << "converged: " << result.converged << std::endl;
    std::cout << "error: " << (result.error / result.num_inliers) << std::endl;
    std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
    std::cout << "--- H ---" << std::endl << result.H << std::endl;
    std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;

    return 0;
}

//// result:
// 0.0486873 -0.998814  -5.83098
//  0.998814 0.0486873  0.987868
//      0         0         1