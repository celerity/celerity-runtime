cmake_minimum_required(VERSION 3.5.1)
project(celerity_full_stack_example)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake )

# Windows-specific:
#
# install libraries with vcpkg using the x64-windows triplet e.g.:
#
#  .\vcpkg.exe install boost-graph:x64-windows
#
# provide CMake toolchain file through -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg_toolchain_file>
#
# the vcpkg tool chain file can be found in:
#
#  <vcpkg_root>/scripts/buildsystems/vcpkg.cmake
#
# either set it as command line parameter when invoking CMake
#
#  OR
#
# set VCPKG_ROOT in the CMakeSettings.json when using the Visual Studio CMake
# integration.
# Similarily the ComputeCpp root directory has to be set in order for CMake to
# find it - again, use one of the approaches above to set
# COMPUTECPP_PACKAGE_ROOT_DIR.

find_package(Boost 1.66.0 COMPONENTS graph REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(OpenCL REQUIRED)
find_package(MPI 2.0 REQUIRED)

# linux-specific
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(COMPUTECPP_USER_FLAGS "-stdlib=libc++ -I/usr/include/libcxxabi")
endif()

# Celerity runtime library

if(MSVC)
    # Add includes to library so they show up in generated VS project
    file(GLOB_RECURSE INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
endif()

add_library(
    celerity_runtime
    src/distr_queue.cc
    src/graph_utils.cc
    src/handler.cc
    src/runtime.cc
    ${INCLUDES}
)

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 14)

message(STATUS "Buildtype: ${CMAKE_BUILD_TYPE}")

target_include_directories(
  celerity_runtime
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor
  ${Boost_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIR}
  ${ComputeCpp_INCLUDE_DIRECTORY}
  ${MPI_CXX_INCLUDE_PATH}
)

target_link_libraries(
  celerity_runtime
  PUBLIC
  ${Boost_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
)

add_sycl_to_target(
  celerity_runtime
  ${CMAKE_CURRENT_BINARY_DIR}
  ""
)

set(Celerity_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor
  ${Boost_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIR}
  ${ComputeCpp_INCLUDE_DIRECTORY}
  ${MPI_CXX_INCLUDE_PATH}
)

# full stack example

add_executable(
  full_stack_example
  main.cc
)

set_property(TARGET full_stack_example PROPERTY CXX_STANDARD 14)

target_include_directories(
  full_stack_example
  PUBLIC
  ${Celerity_INCLUDE_DIRS}
)

target_link_libraries(
  full_stack_example
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  full_stack_example
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
)

# Windows-specific, when using Visual Studio's "Open Folder":
#
# add post build step to copy ComputeCpp DLLs to the output directory
#
function(copy_dlls target)

  # This is not set when using standard CMake generation
  if(NOT CMAKE_BUILD_TYPE)
    return()
  endif()

  if (${CMAKE_BUILD_TYPE} MATCHES "Debug" OR
      ${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")

    get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_DEBUG_WE
      ${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} NAME_WE)

    get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY
      ${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} DIRECTORY )

    add_custom_command(TARGET ${target}
    POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY}/${COMPUTECPP_RUNTIME_LIBRARY_DEBUG_WE}.dll"
      $<TARGET_FILE_DIR:${target}>)

  else()

    get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_WE
      ${COMPUTECPP_RUNTIME_LIBRARY} NAME_WE)

    get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY
      ${COMPUTECPP_RUNTIME_LIBRARY} DIRECTORY )

    add_custom_command(TARGET ${target}
    POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY}/${COMPUTECPP_RUNTIME_LIBRARY_WE}.dll"
      $<TARGET_FILE_DIR:${target}>)

  endif()

endfunction(copy_dlls)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  copy_dlls(full_stack_example)
endif()

# tests

enable_testing(true)
add_subdirectory(test)


