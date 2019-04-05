cmake_minimum_required(VERSION 3.5.1)
project(celerity_runtime)

set(Celerity_VERSION 0.0.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED)
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)
include(Download_spdlog)

# Find SYCL implementation.
# We first check if ComputeCpp is found, otherwise we assume hipSYCL.
# TODO: Use findhipSYCL, once that exists.
find_package(ComputeCpp)
if(ComputeCpp_FOUND)
  set(COMPUTECPP_USER_FLAGS ${COMPUTECPP_USER_FLAGS} -D_CRT_SECURE_NO_WARNINGS)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --restrict-device-header-path=\"${PROJECT_SOURCE_DIR}/include\"")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --restrict-device-header-path=\"${PROJECT_SOURCE_DIR}/test\"")

  # This is a dummy-implementation that doesn't do a whole lot, mainly so
  # we can always assume that this function exists.
  # Note that it does set some target properties to enable compatiblity
  # with hipSYCL (CUDA/Clang).
  function(add_sycl_to_target)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ADD_HIPSYCL
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
    )

    # Disable gnu++14 (breaks CUDA/Clang compilation of libstdc++)
    set_property(TARGET ${ADD_HIPSYCL_TARGET} PROPERTY CXX_EXTENSIONS OFF)
    # Since hipSYCL transforms source files and stores them in a separate location, we have to
    # make sure any local includes can still be found.
    target_include_directories(${ADD_HIPSYCL_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  endfunction()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

if(MSVC)
  # Add includes to library so they show up in generated VS project
  file(GLOB_RECURSE INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
endif()

add_library(
  celerity_runtime
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
  ${INCLUDES}
)

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 14)

target_include_directories(
  celerity_runtime
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor>
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
  SOURCES ""
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
  DESTINATION include
)
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/vendor/allscale/
  DESTINATION include/allscale
)
install(
  FILES ${PROJECT_SOURCE_DIR}/vendor/ctpl.h
  DESTINATION include
)
install(
  DIRECTORY ${spdlog_INCLUDE_DIRS}/spdlog/
  DESTINATION include/spdlog
)
install(
  TARGETS celerity_runtime
  EXPORT install_exports
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
export(
  EXPORT install_exports
  FILE "${CMAKE_CURRENT_BINARY_DIR}/celerity-targets.cmake"
  NAMESPACE Celerity::
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
install(
  FILES
    ${PROJECT_SOURCE_DIR}/cmake/celerity-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/celerity-config-version.cmake
  DESTINATION lib/cmake
)

