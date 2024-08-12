cmake_minimum_required(VERSION 3.10)

# 项目名称
project(GICPExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用位置独立代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 Eigen 库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 查找源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(GICPExample ${SOURCES} main.cpp)

# 添加头文件目录
target_include_directories(GICPExample PRIVATE include)

# 添加 Eigen 头文件目录
target_include_directories(GICPExample PRIVATE ${EIGEN3_INCLUDE_DIR})

# 链接 OpenCV
target_link_libraries(GICPExample PRIVATE ${OpenCV_LIBS})