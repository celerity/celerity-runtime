cmake_minimum_required(VERSION 3.13)

file(STRINGS "VERSION" Celerity_VERSION)
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${Celerity_VERSION}")
set(CELERITY_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CELERITY_VERSION_MINOR ${CMAKE_MATCH_2})
set(CELERITY_VERSION_PATCH ${CMAKE_MATCH_3})
message(VERBOSE "Celerity version is ${CELERITY_VERSION_MAJOR}.${CELERITY_VERSION_MINOR}.${CELERITY_VERSION_PATCH}")

project(celerity_runtime VERSION ${Celerity_VERSION} LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CELERITY_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CELERITY_CMAKE_DIR}")
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)

set(CELERITY_SYCL_IMPL "OFF" CACHE STRING "hipSYCL|ComputeCpp|DPC++")

include("${CELERITY_CMAKE_DIR}/TestDPCXX.cmake")
test_cxx_compiler_is_dpcpp(CXX_COMPILER_IS_DPCPP)
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
    message(STATUS "Found ComputeCpp ${ComputeCpp_VERSION}: ${ComputeCpp_DIR}")
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

if(CELERITY_SYCL_IMPL STREQUAL "ComputeCpp")
  # Determining the compiler version would usually be a job for FindComputeCpp, but since we're vendoring it, we try
  # to avoid introducing unnecessary changes.
  # TODO Replace this once FindComputeCpp supports distinguishing old and new ("experimental") compilers natively.

  execute_process(
    COMMAND "${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}" --version
    RESULT_VARIABLE COMPUTECPP_COMPILER_STATUS
    OUTPUT_VARIABLE COMPUTECPP_COMPILER_VERSION_STRING)
  if(NOT COMPUTECPP_COMPILER_STATUS EQUAL 0)
    message(FATAL_ERROR "Could not invoke ComputeCpp compiler at ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}")
  endif()

  string(REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" COMPUTECPP_COMPILER_VERSION
    "${COMPUTECPP_COMPILER_VERSION_STRING}")
  if(NOT COMPUTECPP_COMPILER_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
    message(FATAL_ERROR "Could not determine ComputeCpp version")
  endif()
  message(STATUS "Your installation of ComputeCpp is based on Clang ${COMPUTECPP_COMPILER_VERSION}")

  list(APPEND CELERITY_CXX_FLAGS -DSYCL_LANGUAGE_VERSION=COMPUTECPP_SYCL_VERSION_2020)
  if(COMPUTECPP_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -sycl-std=2020)
  else()
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -DSYCL_LANGUAGE_VERSION=COMPUTECPP_SYCL_VERSION_2020)
  endif()
endif()

if(MSVC)
  list(APPEND CELERITY_CXX_FLAGS -D_CRT_SECURE_NO_WARNINGS)
endif()

if(CELERITY_SYCL_IMPL STREQUAL hipSYCL)
  macro(test_hipsycl_feature VAR NAME FEATURE)
    try_compile("${VAR}"
      "${PROJECT_BINARY_DIR}/hipsycl_feature_test" # bindir
      "${PROJECT_SOURCE_DIR}/cmake/hipsycl_feature_test" # srcdir
      hipsycl_test # projectName
      "${NAME}" # targetName
      CMAKE_FLAGS "-DhipSYCL_DIR=${hipSYCL_DIR}"
    )
    if("${${VAR}}")
      message(STATUS "hipSYCL supports ${FEATURE}")
    else()
      message(STATUS "hipSYCL does NOT support ${FEATURE}")
    endif()
  endmacro()
  test_hipsycl_feature(HIPSYCL_SUPPORTS_SYCL_2020_REDUCTIONS hipsycl_reduction_test "SYCL 2020 reductions")
endif()

if(CELERITY_SYCL_IMPL STREQUAL hipSYCL AND HIPSYCL_SUPPORTS_SYCL_2020_REDUCTIONS)
  set(CELERITY_FEATURE_SCALAR_REDUCTIONS ON)
  set(CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS ON)
elseif(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  set(CELERITY_FEATURE_SCALAR_REDUCTIONS OFF)
  set(CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS ON)
else()
  set(CELERITY_FEATURE_SCALAR_REDUCTIONS OFF)
  set(CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS OFF)
endif()

set(CELERITY_FEATURE_LOCAL_ACCESSOR ON)

if(NOT CELERITY_SYCL_IMPL STREQUAL ComputeCpp)
  set(CELERITY_FEATURE_UNNAMED_KERNELS ON)
else()
  set(CELERITY_FEATURE_UNNAMED_KERNELS OFF)
endif()

set(CELERITY_RUNTIME_LIBRARY celerity_runtime)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/AddToTarget.cmake")

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

configure_file(include/version.h.in include/version.h @ONLY)

# Add includes to library so they show up in IDEs
file(GLOB_RECURSE INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
list(APPEND INCLUDES "${CMAKE_CURRENT_BINARY_DIR}/include/version.h")

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
  "${CMAKE_CURRENT_BINARY_DIR}/src/version.cc"
)

if(WIN32)
  set(SOURCES ${SOURCES} src/platform_specific/affinity.win.cc)
 elseif(UNIX)
  set(SOURCES ${SOURCES} src/platform_specific/affinity.unix.cc)
endif()

add_library(
  celerity_runtime
  STATIC
  ${SOURCES}
  ${INCLUDES}
)

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 17)

target_include_directories(celerity_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
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
  CELERITY_FEATURE_SCALAR_REDUCTIONS=$<BOOL:${CELERITY_FEATURE_SCALAR_REDUCTIONS}>
  CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS=$<BOOL:${CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS}>
  CELERITY_FEATURE_LOCAL_ACCESSOR=$<BOOL:${CELERITY_FEATURE_LOCAL_ACCESSOR}>
  CELERITY_FEATURE_UNNAMED_KERNELS=$<BOOL:${CELERITY_FEATURE_UNNAMED_KERNELS}>
)

# Collect version information from git in src/version.cc. This target is always out of date, but the timestamp
# (and contents) of version.cc will only change when the HEAD moves or the dirty-status changes. When they do, all
# targets depending on celerity_runtime will have to be re-linked.
add_custom_target(
  celerity_version
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/src/version.cc"
  COMMENT "Generating src/version.cc"
  COMMAND "${CMAKE_COMMAND}"
    -D "CELERITY_SOURCE_DIR=${CMAKE_CURRENT_LIST_DIR}"
    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/GenerateVersionSource.cmake"
  DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/cmake/GenerateVersionSource.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/version.cc.in"
  VERBATIM
)
add_dependencies(celerity_runtime celerity_version)

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

target_compile_options(celerity_runtime PUBLIC "${CELERITY_CXX_FLAGS}")

# Examples

option(CELERITY_BUILD_EXAMPLES "Build various example applications" ON)
if(CELERITY_BUILD_EXAMPLES)
  add_subdirectory(examples)
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
configure_package_config_file(
  vendor/spdlog/cmake/spdlogConfig.cmake.in
  "${spdlog_PROJECT_CONFIG_OUT}"
  INSTALL_DESTINATION lib/celerity/vendor/cmake
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
  PATTERN *.in EXCLUDE
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
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
  DESTINATION lib/cmake/Celerity
)
set(CELERITY_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/celerity-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/celerity-config.cmake"
  @ONLY
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/celerity-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/celerity-config-version.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/AddToTarget.cmake"
  DESTINATION lib/cmake/Celerity
)

if(CELERITY_SYCL_IMPL STREQUAL ComputeCpp)
  install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/FindComputeCpp.cmake"
    DESTINATION lib/cmake/Celerity
  )
elseif(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/TestDPCXX.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/sycl_test.cpp"
    DESTINATION lib/cmake/Celerity
  )
endif()
