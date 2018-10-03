cmake_minimum_required(VERSION 3.5.1)
project(celerity_runtime)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(Boost 1.66.0 COMPONENTS graph REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)
include(Download_spdlog)

add_subdirectory(vendor/ccto)
set_property(
  TARGET ccto ccto_test cl_rect_update_benchmark cl_rect_update_test
  PROPERTY FOLDER "vendor"
)
if(MSVC)
  set_property(TARGET cl_rect_update_lib.headers PROPERTY FOLDER "vendor")
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
  src/buffer.cc
  src/buffer_state.cc
  src/buffer_transfer_manager.cc
  src/distr_queue.cc
  src/graph_builder.cc
  src/graph_generator.cc
  src/graph_utils.cc
  src/handler.cc
  src/runtime.cc
  src/transformers/naive_split.cc
  src/worker_job.cc
  ${INCLUDES}
)

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 14)

target_include_directories(
  celerity_runtime
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor
  ${Boost_INCLUDE_DIRS}
  ${ComputeCpp_INCLUDE_DIRECTORY}
  ${MPI_CXX_INCLUDE_PATH}
  ${spdlog_INCLUDE_DIRS}
)

target_link_libraries(
  celerity_runtime
  PUBLIC
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  ccto
)

add_sycl_to_target(
  TARGET celerity_runtime
  SOURCES ""
)

if(MSVC)
  target_compile_options(celerity_runtime PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(celerity_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter)
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
