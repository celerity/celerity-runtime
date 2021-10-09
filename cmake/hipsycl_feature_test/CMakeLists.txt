cmake_minimum_required(VERSION 3.13)
project(hipsycl_feature_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(HIPSYCL_TARGETS omp)

find_package(hipSYCL REQUIRED CONFIG)

add_executable(hipsycl_reduction_test hipsycl_reduction_test.cpp)
add_sycl_to_target(TARGET hipsycl_reduction_test SOURCES hipsycl_reduction_test.cpp)
