cmake_minimum_required(VERSION 3.13)

set(Celerity_VERSION 0.2.1)
project(celerity_runtime VERSION ${Celerity_VERSION})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")
find_package(Boost 1.65.0 REQUIRED COMPONENTS atomic container)
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)

option(CELERITY_SYCL_IMPL "hipSYCL|ComputeCpp")

# Find SYCL implementation.
set(hipSYCL_MIN_VERSION "0.9.0")
if(CELERITY_SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL ${hipSYCL_MIN_VERSION} CONFIG REQUIRED)
elseif(CELERITY_SYCL_IMPL STREQUAL "ComputeCpp")
  find_package(ComputeCpp REQUIRED)
else()
  # We first check if hipSYCL can be found, otherwise we try ComputeCpp.
  find_package(hipSYCL ${hipSYCL_MIN_VERSION} CONFIG)
  if(hipSYCL_FOUND)
    set(CELERITY_SYCL_IMPL "hipSYCL")
  else()
    find_package(ComputeCpp REQUIRED)
    set(CELERITY_SYCL_IMPL "ComputeCpp")
  endif()
endif()

if(CELERITY_SYCL_IMPL STREQUAL "hipSYCL")
  message(STATUS "Found hipSYCL: ${hipSYCL_DIR}")
else()
  message(STATUS "Found ComputeCpp: ${ComputeCpp_DIR}")
endif()

if(CELERITY_SYCL_IMPL STREQUAL "ComputeCpp" AND NOT COMPUTECPP_USER_FLAGS MATCHES "-D_CRT_SECURE_NO_WARNINGS")
  set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS};-D_CRT_SECURE_NO_WARNINGS=1"
    CACHE STRING "" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

# 3rdparty dependencies
macro(add_submodule_directory RELPATH)
	if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${RELPATH}/CMakeLists.txt")
		message(SEND_ERROR "${RELPATH} is missing, perform a git submodule update")
	endif()
	add_subdirectory("${RELPATH}")
endmacro()

add_submodule_directory(vendor/spdlog)
add_submodule_directory(vendor/Catch2)

# Add includes to library so they show up in IDEs
file(GLOB_RECURSE INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

if(CMAKE_GENERATOR STREQUAL "Ninja")
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
endif()

set(SOURCES
  src/buffer_manager.cc
  src/buffer_storage.cc
  src/buffer_transfer_manager.cc
  src/command_graph.cc
  src/config.cc
  src/device_queue.cc
  src/executor.cc
  src/graph_generator.cc
  src/graph_serializer.cc
  src/mpi_support.cc
  src/print_graph.cc
  src/print_utils.cc
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

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 17)

target_include_directories(celerity_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/celerity>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor>
  $<INSTALL_INTERFACE:include/celerity/vendor>
  ${Boost_INCLUDE_DIRS}
)

target_link_libraries(celerity_runtime PUBLIC
  Threads::Threads
  ${Boost_LIBRARIES}
  MPI::MPI_CXX
  spdlog::spdlog
)

# For debug builds, we set the CELERITY_DETAIL_ENABLE_DEBUG preprocessor flag,
# which allows Celerity to control debug functionality within headers regardless
# of a user target's build type. (This flag is not intended to be modified by
# end users directly).
#
# To make this work, we need to ensure that user targets also receive this flag
# whenever they link to a Celerity runtime that was built with the DEBUG
# configuration. Unfortunately there doesn't seem to be a way of doing this using
# generator expressions, which is why we have to do it manually within
# celerity-config.cmake instead.
target_compile_definitions(celerity_runtime PUBLIC
  # We still mark this as PUBLIC during builds (but not installation),
  # so that the examples and tests receive the correct flag as well.
  $<BUILD_INTERFACE:
    $<$<CONFIG:Debug>:CELERITY_DETAIL_ENABLE_DEBUG>
  >
)

if(CELERITY_SYCL_IMPL STREQUAL "ComputeCpp")
	target_link_libraries(celerity_runtime PUBLIC ComputeCpp::ComputeCpp)
endif()

set(DEVICE_SOURCES "")
if(CELERITY_SYCL_IMPL STREQUAL "hipSYCL")
  # For hipSYCL we have to pass all source files into add_sycl_to_target
  # Don't just do it in general (yields better ComputeCpp build performance)
  set(DEVICE_SOURCES ${SOURCES})
endif()

add_sycl_to_target(
  TARGET celerity_runtime
  SOURCES ${DEVICE_SOURCES}
)

if(MSVC)
  target_compile_options(celerity_runtime PRIVATE /MP /W3 /D_CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(celerity_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter -Werror=return-type -Werror=init-self)
endif()

if(CELERITY_SYCL_IMPL STREQUAL "hipSYCL")
  # Boost currently (as of 1.71) does not enable variadic macros
  # when it detects CUDA compilation.
  # Since we are using Clang however (via hipSYCL) instead of NVCC,
  # the macros can and should be enabled.
  # See https://github.com/boostorg/preprocessor/issues/24
  target_compile_definitions(celerity_runtime PUBLIC BOOST_PP_VARIADICS=1)
endif()

# Examples

option(CELERITY_BUILD_EXAMPLES "Build various example applications" ON)
if(CELERITY_BUILD_EXAMPLES)
  add_subdirectory(examples/convolution)
  add_subdirectory(examples/matmul)
  add_subdirectory(examples/syncing)
  add_subdirectory(examples/wave_sim)

  set_property(
    TARGET convolution matmul syncing wave_sim
    PROPERTY FOLDER "examples"
  )

  find_package(PkgConfig)
  if (PkgConfig_FOUND)
	pkg_check_modules(HDF5 IMPORTED_TARGET hdf5-1.12.0)
	if(HDF5_FOUND)
      add_subdirectory(examples/distr_io)
      set_property(TARGET distr_io PROPERTY FOLDER "examples")
    endif()
  endif()
endif()

# Tests

enable_testing(true)
add_subdirectory(test)

# Install

include(CMakePackageConfigHelpers)

# Install 3rdparty spdlog under celerity/vendor
set(config_targets_file spdlogConfigTargets.cmake) # This variable is used in spdlogConfig.cmake.in
set(spdlog_PROJECT_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/vendor/spdlog/spdlogConfig.cmake")
set(spdlog_VERSION_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/vendor/spdlog/spdlogConfigVersion.cmake")
install(
	DIRECTORY vendor/spdlog/include/spdlog/
	DESTINATION include/celerity/vendor/spdlog
)
install(
	TARGETS spdlog
	EXPORT spdlog
	DESTINATION lib/celerity/vendor
)
install(
	EXPORT spdlog
	DESTINATION lib/celerity/vendor/cmake
	NAMESPACE spdlog::
	FILE spdlogConfigTargets.cmake
)
configure_file(
	vendor/spdlog/cmake/spdlogConfig.cmake.in
	"${spdlog_PROJECT_CONFIG_OUT}"
	@ONLY
)
write_basic_package_version_file(
	"${spdlog_VERSION_CONFIG_FILE}"
	VERSION ${spdlog_VERSION}
	COMPATIBILITY SameMajorVersion
)
install(FILES "${spdlog_PROJECT_CONFIG_OUT}" "${spdlog_VERSION_CONFIG_FILE}"
	DESTINATION lib/celerity/vendor/cmake
)

# Install celerity
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include/celerity
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
  TARGETS celerity_runtime
  EXPORT install_exports
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
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

