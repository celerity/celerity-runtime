cmake_minimum_required(VERSION 3.8)
project(celerity_full_stack_example)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(FindComputeCpp)
find_package(Boost 1.66.0 REQUIRED)

add_executable(full_stack_example main.cc celerity_runtime.h celerity_runtime.cc)
target_include_directories(full_stack_example PUBLIC ${Boost_INCLUDE_DIRS} ${COMPUTECPP_INCLUDE_DIRECTORY})
#set_property(TARGET full_stack_example PROPERTY CXX_STANDARD 17) # THIS MAY CAUSE ISSUES
add_sycl_to_target(full_stack_example ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
