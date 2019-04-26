cmake_minimum_required(VERSION 3.5.1)
project(celerity_runtime)

set(Celerity_VERSION 0.0.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED)
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)

# Find SYCL implementation.
# We first check if hipSYCL can be found, otherwise we try ComputeCpp.
set(HIPSYCL_RESTRICT_DEVICE_HEADER_PATH "${PROJECT_SOURCE_DIR}/include;${PROJECT_SOURCE_DIR}/test")
find_package(hipSYCL CONFIG)
if(NOT hipSYCL_FOUND)
  find_package(ComputeCpp REQUIRED)
  set(
    COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS} -D_CRT_SECURE_NO_WARNINGS"
    CACHE STRING "" FORCE
  )
endif()

include(Download_spdlog)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

if(MSVC)
  # Add includes to library so they show up in generated VS project
  file(GLOB_RECURSE INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
endif()

set(SOURCES
  src/buffer_transfer_manager.cc
  src/config.cc
  src/device_queue.cc
  src/executor.cc
  src/graph_builder.cc
  src/graph_generator.cc
  src/graph_utils.cc
  src/mpi_support.cc
  src/runtime.cc
  src/scheduler.cc
  src/task.cc
  src/task_manager.cc
  src/transformers/naive_split.cc
  src/user_bench.cc
  src/worker_job.cc
)

add_library(
  celerity_runtime
  STATIC
  ${SOURCES}
  ${INCLUDES}
)

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 14)

target_include_directories(
  celerity_runtime
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/celerity>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor>
  $<INSTALL_INTERFACE:include/celerity/vendor>
  ${Boost_INCLUDE_DIRS}
  ${ComputeCpp_INCLUDE_DIRECTORY}
  ${MPI_CXX_INCLUDE_PATH}
  $<BUILD_INTERFACE:${spdlog_INCLUDE_DIRS}>
)

target_link_libraries(
  celerity_runtime
  PUBLIC
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
)

add_sycl_to_target(
  TARGET celerity_runtime
  SOURCES "${SOURCES}"
)

if(MSVC)
  target_compile_options(celerity_runtime PRIVATE /MP /W3 /D_CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(celerity_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter -D_CRT_SECURE_NO_WARNINGS)
endif()

# Examples

option(CELERITY_BUILD_EXAMPLES "Build various example applications" ON)
if(CELERITY_BUILD_EXAMPLES)
  add_subdirectory(examples/simple)
  add_subdirectory(examples/gaussian)
  add_subdirectory(examples/matmul)
  add_subdirectory(examples/wave_sim)

  set_property(
    TARGET simple gaussian matmul wave_sim
    PROPERTY FOLDER "examples"
  )
endif()

# Tests

enable_testing(true)
add_subdirectory(test)

# Install

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include/celerity/celerity
)
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/vendor/allscale/
  DESTINATION include/celerity/vendor/allscale
)
install(
  FILES ${PROJECT_SOURCE_DIR}/vendor/ctpl.h
  DESTINATION include/celerity/vendor
)
install(
  DIRECTORY ${spdlog_INCLUDE_DIRS}/spdlog/
  DESTINATION include/celerity/vendor/spdlog
)
install(
  TARGETS celerity_runtime
  EXPORT install_exports
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/celerity-config-version.cmake"
  VERSION ${Celerity_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT install_exports
  FILE "celerity-targets.cmake"
  NAMESPACE Celerity::
  DESTINATION lib/cmake
)
set(CELERITY_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
if(hipSYCL_FOUND)
  set(CELERITY_SYCL_IMPL "hipSYCL")
else()
  set(CELERITY_SYCL_IMPL "ComputeCpp")
endif()
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/celerity-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/celerity-config.cmake
  @ONLY
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/celerity-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/celerity-config-version.cmake
  DESTINATION lib/cmake
)

