cmake_minimum_required(VERSION 3.13)

if (CMAKE_CURRENT_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "Celerity does not support in-source builds.\nPlease use a dedicated build directory and remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt and ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
endif()

file(STRINGS "VERSION" Celerity_VERSION)
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${Celerity_VERSION}")
set(CELERITY_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CELERITY_VERSION_MINOR ${CMAKE_MATCH_2})
set(CELERITY_VERSION_PATCH ${CMAKE_MATCH_3})
message(VERBOSE "Celerity version is ${CELERITY_VERSION_MAJOR}.${CELERITY_VERSION_MINOR}.${CELERITY_VERSION_PATCH}")

project(celerity_runtime VERSION ${Celerity_VERSION} LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEFAULT_ENABLE_DEBUG_CHECKS ON)
else()
  set(DEFAULT_ENABLE_DEBUG_CHECKS OFF)
endif()

option(CELERITY_ACCESS_PATTERN_DIAGNOSTICS "Diagnose uninitialized reads and overlapping writes" ${DEFAULT_ENABLE_DEBUG_CHECKS})
option(CELERITY_ACCESSOR_BOUNDARY_CHECK "Enable accessor boundary check" ${DEFAULT_ENABLE_DEBUG_CHECKS})

if(CELERITY_ACCESSOR_BOUNDARY_CHECK AND NOT (CMAKE_BUILD_TYPE STREQUAL "Debug"))
  message(STATUS "Accessor boundary check enabled - this will impact kernel performance")
endif()

set(CELERITY_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CELERITY_CMAKE_DIR}")
find_package(Threads REQUIRED)

option(CELERITY_ENABLE_MPI "Enable MPI support" ON)
if(CELERITY_ENABLE_MPI)
  find_package(MPI 2.0 REQUIRED)
  message(STATUS "Found MPI: ${MPI_CXX_COMPILER}")
else()
  message(STATUS "Building without MPI support (disabled)")
endif()

set(CELERITY_SYCL_IMPL "OFF" CACHE STRING "AdaptiveCpp|DPC++|SimSYCL")

if(CELERITY_SYCL_IMPL STREQUAL "hipSYCL")
  set(CELERITY_SYCL_IMPL "AdaptiveCpp" CACHE STRING "" FORCE)
endif()

include("${CELERITY_CMAKE_DIR}/TestDPCXX.cmake")
test_cxx_compiler_is_dpcpp(CXX_COMPILER_IS_DPCPP)
if(CXX_COMPILER_IS_DPCPP)
  message(STATUS "CXX compiler at ${CMAKE_CXX_COMPILER} detected as DPC++")
endif()

# Find SYCL implementation.
if(CELERITY_SYCL_IMPL STREQUAL "AdaptiveCpp")
  find_package(AdaptiveCpp CONFIG REQUIRED)
elseif(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  if(NOT CXX_COMPILER_IS_DPCPP)
    message(FATAL_ERROR "CELERITY_SYCL_IMPL set to DPC++, but CXX compiler at ${CMAKE_CXX_COMPILER} is not DPC++")
  endif()
elseif(CELERITY_SYCL_IMPL STREQUAL "SimSYCL")
  find_package(SimSYCL CONFIG REQUIRED)
elseif(NOT "${CELERITY_SYCL_IMPL}x" STREQUAL "OFFx")
  message(FATAL_ERROR "Invalid SYCL implementation ${CELERITY_SYCL_IMPL} specified")
else()
  if(CXX_COMPILER_IS_DPCPP)
    list(APPEND AVAILABLE_SYCL_IMPLS DPC++)
  endif()

  find_package(AdaptiveCpp QUIET CONFIG)
  if(AdaptiveCpp_FOUND)
    message(STATUS "Found AdaptiveCpp: ${AdaptiveCpp_DIR}")
    list(APPEND AVAILABLE_SYCL_IMPLS AdaptiveCpp)
  endif()

  find_package(SimSYCL QUIET CONFIG)
  if(SimSYCL_FOUND)
    message(STATUS "Found SimSYCL: ${SimSYCL_DIR}")
    list(APPEND AVAILABLE_SYCL_IMPLS SimSYCL)
  endif()

  list(LENGTH AVAILABLE_SYCL_IMPLS NUM_AVAILABLE_SYCL_IMPLS)
  if(NUM_AVAILABLE_SYCL_IMPLS EQUAL 0)
    message(FATAL_ERROR "No SYCL implementation found. You might need to add an install path for AdaptiveCpp "
    "or SimSYCL to CMAKE_PREFIX_PATH.")
  elseif(NUM_AVAILABLE_SYCL_IMPLS GREATER 1)
    list(JOIN AVAILABLE_SYCL_IMPLS " and " AVAILABLE_SYCL_IMPLS_STR)
    message(FATAL_ERROR "More than one SYCL implementation available: Found ${AVAILABLE_SYCL_IMPLS_STR}. "
      "Please choose one implementation using -DCELERITY_SYCL_IMPL=AdaptiveCpp|DPC++|SimSYCL.")
  else()
    set(CELERITY_SYCL_IMPL "${AVAILABLE_SYCL_IMPLS}")
    message(STATUS "Automatically choosing ${CELERITY_SYCL_IMPL} because it is the only SYCL implementation available")
  endif()
endif()

set(CELERITY_DPCPP_TARGETS "spir64" CACHE STRING "Intel DPC++ targets")
if(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  message(STATUS "DPC++ will target ${CELERITY_DPCPP_TARGETS}")
endif()

if(WIN32)
  list(APPEND CELERITY_CXX_FLAGS -D_CRT_SECURE_NO_WARNINGS)
endif()

set(CELERITY_MINIMUM_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD "${CELERITY_MINIMUM_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CELERITY_RUNTIME_LIBRARY celerity_runtime)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/AddToTarget.cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

if(CELERITY_SYCL_IMPL STREQUAL "AdaptiveCpp")
  list(APPEND CELERITY_CXX_FLAGS -DACPP_ALLOW_INSTANT_SUBMISSION=1)
endif()

if(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  # See https://github.com/oneapi-src/unified-runtime/issues/803
  message(STATUS "Not enabling mimalloc by default because it breaks with oneAPI plugin loading")
  set(CELERITY_USE_MIMALLOC_DEFAULT OFF)
elseif(CELERITY_SYCL_IMPL STREQUAL "SimSYCL" AND SIMSYCL_ENABLE_ASAN)
  message(STATUS "Not enabling mimalloc by default because SimSYCL is built with AddressSanitizer")
  set(CELERITY_USE_MIMALLOC_DEFAULT OFF)
  list(APPEND CELERITY_CXX_FLAGS -fsanitize=address)
  list(APPEND CELERITY_LINK_FLAGS -fsanitize=address)
else()
  set(CELERITY_USE_MIMALLOC_DEFAULT ON)
endif()
option(CELERITY_USE_MIMALLOC "Use the mimalloc memory allocator" ${CELERITY_USE_MIMALLOC_DEFAULT})

option(CELERITY_TRACY_SUPPORT "Enable Tracy profiler integration" OFF)

# 3rdparty dependencies
include(FetchContent)

macro(fetch_content_from_submodule DEPNAME RELPATH)
  # Dependency might already have been satisfied previously by
  #   - the top-level CMake project including celerity-runtime as a dependency, but not its first one
  #   - another depencency of celerity-runtime that it transitively shares a depencency with (SimSYCL -> libenvpp)
  # The top-level CMake project must ensure that there are no conflicts between dependency versions in this case.
  if(${DEPNAME}_FOUND)
    message(STATUS "Using existing ${DEPNAME} in ${${DEPNAME}_DIR}, please ensure that the version (${${DEPNAME}_VERSION}) is compatible")
    set(CELERITY_DETAIL_USE_SYSTEM_${DEPNAME} ON)
  else()
    message(STATUS "Fetching ${DEPNAME} from submodule")
    set(CELERITY_DETAIL_USE_SYSTEM_${DEPNAME} OFF)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${RELPATH}/CMakeLists.txt")
      message(FATAL_ERROR "The git submodule ${RELPATH} is missing.\nTry running `git submodule update --init`.")
    endif()
    FetchContent_Declare(${DEPNAME}
      URL "${PROJECT_SOURCE_DIR}/${RELPATH}"
    )
    FetchContent_MakeAvailable(${DEPNAME})
  endif()
endmacro()

set(FMT_INSTALL ON CACHE BOOL "" FORCE)
fetch_content_from_submodule(fmt vendor/fmt)

set(SPDLOG_INSTALL ON CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
fetch_content_from_submodule(spdlog vendor/spdlog)

fetch_content_from_submodule(small_vector vendor/small_vector)

if(WIN32)
  # Catch2 uses custom top-level exception filters that conflict with MSMPI, which also uses their own filters.
  # Disabling Windows SEH prevents Catch2 from using their own top-level exception filters.
  set(CATCH_CONFIG_NO_WINDOWS_SEH ON)
endif()
fetch_content_from_submodule(Catch2 vendor/Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

set(LIBENVPP_INSTALL ON CACHE BOOL "" FORCE)
fetch_content_from_submodule(libenvpp vendor/libenvpp)

if(CELERITY_USE_MIMALLOC)
  set(MI_OVERRIDE ON CACHE BOOL "" FORCE)
  set(MI_BUILD_SHARED ON CACHE BOOL "" FORCE)
  set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  fetch_content_from_submodule(mimalloc vendor/mimalloc)
endif()

if(CELERITY_TRACY_SUPPORT)
  set(TRACY_FIBERS ON CACHE BOOL "" FORCE)
  set(TRACY_STATIC ON CACHE BOOL "" FORCE)
  set(TRACY_NO_SAMPLING ON CACHE BOOL "" FORCE)
  set(TRACY_NO_SYSTEM_TRACING ON CACHE BOOL "" FORCE)

  set(PREVIOUS_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Disable warning until we have https://github.com/wolfpld/tracy/pull/924
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
  endif()
  fetch_content_from_submodule(Tracy vendor/tracy)
  set(CMAKE_CXX_FLAGS "${PREVIOUS_CMAKE_CXX_FLAGS}")
endif()

# Deprecated feature flags
set(CELERITY_FEATURE_SCALAR_REDUCTIONS ON)
set(CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS ON)
set(CELERITY_FEATURE_LOCAL_ACCESSOR ON)
set(CELERITY_FEATURE_UNNAMED_KERNELS ON)

# Add header files to library so they show up in IDEs
file(GLOB_RECURSE ALL_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl")

if(CMAKE_GENERATOR STREQUAL "Ninja")
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

# TODO move all non-public headers from "include" to "src"
set(PUBLIC_HEADERS
  include/accessor.h
  include/buffer.h
  include/celerity.h
  include/cgf_diagnostics.h
  include/cgf.h
  include/closure_hydrator.h
  include/config.h
  include/debug.h
  include/device_selector.h
  include/distr_queue.h
  include/fence.h
  include/grid.h
  include/handler.h
  include/hint.h
  include/host_object.h
  include/host_utils.h
  include/item.h
  include/log.h
  include/partition.h
  include/print_utils.h
  include/queue.h
  include/range_mapper.h
  include/ranges.h
  include/reduction.h
  include/runtime.h
  include/side_effect.h
  include/sycl_wrappers.h
  include/tracy.h
  include/types.h
  include/utils.h
  include/workaround.h
)

set(SOURCES
  src/affinity.cc
  src/config.cc
  src/command_graph_generator.cc
  src/dry_run_executor.cc
  src/grid.cc
  src/instruction_graph_generator.cc
  src/live_executor.cc
  src/named_threads.cc
  src/out_of_order_engine.cc
  src/print_graph.cc
  src/recorders.cc
  src/receive_arbiter.cc
  src/runtime.cc
  src/scheduler.cc
  src/split.cc
  src/task.cc
  src/task_manager.cc
  src/utils.cc
  "${CMAKE_CURRENT_BINARY_DIR}/src/version.cc"
)

if(CELERITY_ENABLE_MPI)
  set(SOURCES ${SOURCES} src/mpi_communicator.cc)
endif()

set(CELERITY_DETAIL_HAS_NAMED_THREADS OFF)

if(WIN32)
  set(SOURCES ${SOURCES} src/platform_specific/affinity.win.cc)
  set(SOURCES ${SOURCES} src/platform_specific/named_threads.win.cc)
  set(CELERITY_DETAIL_HAS_NAMED_THREADS ON)
elseif(UNIX)
  if(NOT APPLE)
    set(SOURCES ${SOURCES} src/platform_specific/affinity.unix.cc)
    set(CELERITY_DETAIL_HAS_NAMED_THREADS ON)
  endif()
  set(SOURCES ${SOURCES} src/platform_specific/named_threads.unix.cc)
endif()

# Read by configure_file()
set(CELERITY_SYCL_IS_ACPP OFF)
set(CELERITY_SYCL_IS_DPCPP OFF)
set(CELERITY_SYCL_IS_SIMSYCL OFF)
if(CELERITY_SYCL_IMPL STREQUAL "AdaptiveCpp")
  set(CELERITY_SYCL_IS_ACPP ON)
elseif(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  set(CELERITY_SYCL_IS_DPCPP ON)
elseif(CELERITY_SYCL_IMPL STREQUAL "SimSYCL")
  set(CELERITY_SYCL_IS_SIMSYCL ON)
endif()

configure_file(include/version.h.in include/version.h @ONLY)
list(APPEND ALL_INCLUDES "${CMAKE_CURRENT_BINARY_DIR}/include/version.h")
list(APPEND PUBLIC_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/include/version.h")

add_library(
  celerity_runtime
  STATIC
  ${SOURCES}
  ${ALL_INCLUDES}
)

target_include_directories(celerity_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/celerity>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/matchbox/include>
  $<INSTALL_INTERFACE:include/celerity/vendor>
)

# With DPC++ on windows the correct release (sycl) / debug (sycld) library needs to be linked,
# because they are not ABI compatible.
if(WIN32 AND CELERITY_SYCL_IMPL STREQUAL "DPC++")
  set(SYCL_LIB
    $<$<CONFIG:Debug>:sycl6d>
    $<$<CONFIG:Release>:sycl6>
  )
elseif(CELERITY_SYCL_IMPL STREQUAL SimSYCL)
  set(SYCL_LIB SimSYCL::simsycl)
endif()

if(CELERITY_USE_MIMALLOC)
  target_link_libraries(celerity_runtime PUBLIC mimalloc)
endif()

if(CELERITY_TRACY_SUPPORT)
  target_link_libraries(celerity_runtime PUBLIC Tracy::TracyClient)
endif()

target_link_libraries(celerity_runtime PUBLIC
  Threads::Threads
  fmt::fmt
  spdlog::spdlog
  gch::small_vector
  libenvpp::libenvpp
  ${SYCL_LIB}
)

if(CELERITY_ENABLE_MPI)
  target_link_libraries(celerity_runtime PUBLIC MPI::MPI_CXX)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/backend)
target_link_libraries(celerity_runtime PUBLIC celerity_backends)

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
    CELERITY_DETAIL_ENABLE_DEBUG=$<CONFIG:Debug>
  >
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

add_sycl_to_target(
  TARGET celerity_runtime
  SOURCES ${SOURCES}
)

if(MSVC)
  target_compile_options(celerity_runtime PRIVATE /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(celerity_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter -Werror=return-type -Werror=init-self -Werror=undef)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(celerity_runtime PRIVATE -Wno-array-bounds -Wno-stringop-overflow)  # false positives with GCC 13.2
endif()

target_compile_options(celerity_runtime PUBLIC "${CELERITY_CXX_FLAGS}")
target_link_options(celerity_runtime PUBLIC "${CELERITY_LINK_FLAGS}")

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

# Install celerity
install(
  FILES ${PUBLIC_HEADERS}
  DESTINATION include/celerity
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include/celerity
)
install(
  FILES
    ${PROJECT_SOURCE_DIR}/vendor/matchbox/include/matchbox.hh
  DESTINATION include/celerity/vendor
)
install(
  TARGETS celerity_runtime celerity_backends
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

if(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/TestDPCXX.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/sycl_test.cpp"
    DESTINATION lib/cmake/Celerity
  )
endif()
