cmake_minimum_required(VERSION 3.10)
project(GICPExample)

# 查找 Eigen 库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 查找源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(GICPExample main.cpp ${SOURCES})

# 添加头文件目录
target_include_directories(GICPExample PRIVATE include)

# 添加 Eigen 头文件
target_include_directories(GICPExample PRIVATE ${EIGEN3_INCLUDE_DIR})
