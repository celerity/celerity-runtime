cmake_minimum_required(VERSION 3.5.1)
project(celerity_runtime)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake )

find_package(Boost 1.66.0 COMPONENTS graph REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(OpenCL REQUIRED)
find_package(MPI 2.0 REQUIRED)
find_package(Threads REQUIRED)
include(Download_spdlog)

if(MSVC)
    # Add includes to library so they show up in generated VS project
    file(GLOB_RECURSE INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
endif()

add_library(
    celerity_runtime
    src/buffer_transfer_manager.cc
    src/distr_queue.cc
    src/graph_utils.cc
    src/handler.cc
    src/runtime.cc
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
  ${OpenCL_INCLUDE_DIR}
  ${ComputeCpp_INCLUDE_DIRECTORY}
  ${MPI_CXX_INCLUDE_PATH}
  ${spdlog_INCLUDE_DIRS}
)

target_link_libraries(
  celerity_runtime
  PUBLIC
  ${CMAKE_THREAD_LIBS_INIT}
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
  ${spdlog_INCLUDE_DIRS}
)

option(CELERITY_BUILD_EXAMPLES "Build various example applications" ON)
if(CELERITY_BUILD_EXAMPLES)
  add_subdirectory(examples/simple)
  add_subdirectory(examples/gaussian)
  add_subdirectory(examples/matmul)
endif()

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

