cmake_minimum_required(VERSION 3.5.1)
include(CMakeFindDependencyMacro)

find_dependency(Boost 1.66.0 COMPONENTS graph REQUIRED)
find_dependency(ComputeCpp REQUIRED)
find_dependency(MPI 2.0 REQUIRED)
find_dependency(Threads REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/celerity-targets.cmake")

if(MSVC)
  # Compute the installation path relative to this file.
  # Used for copying DLL below.
  get_filename_component(_CELERITY_INSTALL_PATH "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(_CELERITY_INSTALL_PATH "${_CELERITY_INSTALL_PATH}" PATH)
  get_filename_component(_CELERITY_INSTALL_PATH "${_CELERITY_INSTALL_PATH}" PATH)
  if(_CELERITY_INSTALL_PATH STREQUAL "/")
    set(_CELERITY_INSTALL_PATH "")
  endif()
endif()

function(add_celerity_to_target)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(ADD_CELERITY
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  set_property(
    TARGET ${ADD_CELERITY_TARGET}
    APPEND PROPERTY LINK_LIBRARIES Celerity::celerity_runtime
  )

  add_sycl_to_target(
    TARGET ${ADD_CELERITY_TARGET}
    SOURCES ${ADD_CELERITY_SOURCES}
  )
endfunction()

