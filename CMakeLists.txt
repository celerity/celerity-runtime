cmake_minimum_required(VERSION 3.13)

set(Celerity_VERSION 0.2.1)
project(celerity_runtime VERSION ${Celerity_VERSION} LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)

set(CELERITY_SYCL_IMPL "OFF" CACHE STRING "hipSYCL|ComputeCpp|DPC++")

try_compile(CXX_COMPILER_IS_DPCPP "${PROJECT_BINARY_DIR}/detect_dpcpp"
  SOURCES "${PROJECT_SOURCE_DIR}/cmake/sycl_test.cpp"
  COMPILE_DEFINITIONS -fsycl -sycl-std=2020
  )
if(CXX_COMPILER_IS_DPCPP)
  message(STATUS "CXX compiler at ${CMAKE_CXX_COMPILER} detected as DPC++")
endif()

# Find SYCL implementation.
set(hipSYCL_MIN_VERSION "0.9.1")
if(CELERITY_SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL ${hipSYCL_MIN_VERSION} CONFIG REQUIRED)
elseif(CELERITY_SYCL_IMPL STREQUAL "ComputeCpp")
  find_package(ComputeCpp REQUIRED)
elseif(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  if(NOT CXX_COMPILER_IS_DPCPP)
    message(FATAL_ERROR "CELERITY_SYCL_IMPL set to DPC++, but CXX compiler at ${CMAKE_CXX_COMPILER} is not DPC++")
  endif()
elseif(NOT "${CELERITY_SYCL_IMPL}x" STREQUAL "OFFx")
  message(FATAL_ERROR "Invalid SYCL implementation ${CELERITY_SYCL_IMPL} specified")
else()
  if(CXX_COMPILER_IS_DPCPP)
    list(APPEND AVAILABLE_SYCL_IMPLS DPC++)
  endif()

  find_package(hipSYCL ${hipSYCL_MIN_VERSION} QUIET CONFIG)
  if(hipSYCL_FOUND)
    message(STATUS "Found hipSYCL: ${hipSYCL_DIR}")
    list(APPEND AVAILABLE_SYCL_IMPLS hipSYCL)
  endif()

  find_package(ComputeCpp QUIET)
  if(ComputeCpp_FOUND)
    message(STATUS "Found ComputeCpp: ${ComputeCpp_DIR}")
    list(APPEND AVAILABLE_SYCL_IMPLS ComputeCpp)
  endif()

  list(LENGTH AVAILABLE_SYCL_IMPLS NUM_AVAILABLE_SYCL_IMPLS)
  if(NUM_AVAILABLE_SYCL_IMPLS EQUAL 0)
    message(FATAL_ERROR "No SYCL implementation found. You might need to add an install path for hipSYCL or "
      "ComputeCpp to CMAKE_PREFIX_PATH.")
  elseif(NUM_AVAILABLE_SYCL_IMPLS GREATER 1)
    list(JOIN AVAILABLE_SYCL_IMPLS " and " AVAILABLE_SYCL_IMPLS_STR)
    message(FATAL_ERROR "More than one SYCL implementation available: Found ${AVAILABLE_SYCL_IMPLS_STR}. "
      "Please choose one implementation using -DCELERITY_SYCL_IMPL=hipSYCL|ComputeCpp|DPC++.")
  else()
    set(CELERITY_SYCL_IMPL "${AVAILABLE_SYCL_IMPLS}")
    message(STATUS "Automatically chooosing ${CELERITY_SYCL_IMPL} because it is the only SYCL implementation available")
  endif()
endif ()

set(CELERITY_DPCPP_TARGETS "spir64" CACHE STRING "Intel DPC++ targets")
if(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  message(STATUS "DPC++ will target ${CELERITY_DPCPP_TARGETS}")
endif()

if (CELERITY_SYCL_IMPL STREQUAL "ComputeCpp" AND NOT COMPUTECPP_USER_FLAGS MATCHES "-D_CRT_SECURE_NO_WARNINGS")
    set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS};-D_CRT_SECURE_NO_WARNINGS=1"
            CACHE STRING "" FORCE)
endif ()

# Currently requires a patched hipSYCL (https://github.com/illuhad/hipSYCL/pull/578) to enable
set(CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS FALSE CACHE BOOL "hipSYCL installation supports SYCL 2020 reductions")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif ()

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
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
)

target_link_libraries(celerity_runtime PUBLIC
  Threads::Threads
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
  CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS=$<BOOL:${CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS}>
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

if(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  function(add_sycl_to_target)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args)
    cmake_parse_arguments(ADD_SYCL
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
      )
    # Wno-deprecated-declarations: Kernel offsets are used all over the place
    set(DPCPP_FLAGS "-fsycl;-sycl-std=2020;-fsycl-targets=${CELERITY_DPCPP_TARGETS};-DCELERITY_DPCPP=1;-Wno-deprecated-declarations;${DPCPP_FLAGS}")
    target_compile_options(${ADD_SYCL_TARGET} PUBLIC ${DPCPP_FLAGS})
    target_link_options(${ADD_SYCL_TARGET} PUBLIC ${DPCPP_FLAGS})
  endfunction()
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

# Examples

option(CELERITY_BUILD_EXAMPLES "Build various example applications" ON)
option(CELERITY_EXAMPLES_REQUIRE_HDF5 "Fail on missing HDF5 dependency instead of skipping the distr_io example" OFF)
if(CELERITY_BUILD_EXAMPLES)
  add_subdirectory(examples/convolution)
  add_subdirectory(examples/matmul)
  add_subdirectory(examples/syncing)
  add_subdirectory(examples/wave_sim)

  set_property(
    TARGET convolution matmul syncing wave_sim
    PROPERTY FOLDER "examples"
  )

  if (CELERITY_SYCL_IMPL STREQUAL "hipSYCL" AND CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)
      # CCPP currently does not implement reductions, and DPC++ does not support multiple reduction variables per kernel
      add_subdirectory(examples/reduction)
      set_property(TARGET reduction PROPERTY FOLDER "examples")
  endif ()

  find_package(PkgConfig)
  unset(HDF5_FOUND)
  if(PkgConfig_FOUND)
    pkg_search_module(HDF5 IMPORTED_TARGET hdf5-openmpi hdf5-1.12.0 hdf5)
    if(HDF5_FOUND)
      add_subdirectory(examples/distr_io)
      set_property(TARGET distr_io PROPERTY FOLDER "examples")
    endif()
  endif()
  if(CELERITY_EXAMPLES_REQUIRE_HDF5 AND NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found, but required by CELERITY_EXAMPLES_REQUIRE_HDF5")
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
  FILES ${PROJECT_SOURCE_DIR}/vendor/ctpl_stl.h
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

