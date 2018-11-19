cmake_minimum_required(VERSION 3.5.1)
include(CMakeFindDependencyMacro)

find_dependency(Boost 1.66.0 COMPONENTS graph REQUIRED)
find_dependency(ComputeCpp REQUIRED)
find_dependency(MPI 2.0 REQUIRED)
find_dependency(Threads REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/celerity-targets.cmake")

if(MSVC)
  # FIXME
  # For some reason CMake has trouble finding the ccto library on Windows
  # if the IMPORTED_LOCATION_<CONFIG> property is not set.
  # It does set IMPORTED_IMPLIB_<CONFIG>, but that doesn't seem to be
  # sufficient when trying to link to ccto from an external project.
  #
  # This might be due to the weird setup we have with celerity_runtime and ccto
  # both being exported (using install(EXPORT)) with the same export name.
  #
  # As a workaround we simply set the value ourselves.
  get_target_property(ccto_configs Celerity::ccto IMPORTED_CONFIGURATIONS)
  foreach(ccto_config IN LISTS ccto_configs)
    get_target_property(
      config_location Celerity::ccto IMPORTED_IMPLIB_${ccto_config}
    )
    set_target_properties(
      Celerity::ccto PROPERTIES
      IMPORTED_LOCATION_${ccto_config} config_location
    )
  endforeach()

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
  cmake_parse_arguments(SDK_ADD_CELERITY
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  set_property(
    TARGET ${SDK_ADD_CELERITY_TARGET}
    # Add ccto explicitly to ensure that it is loaded before OpenCL
    APPEND PROPERTY LINK_LIBRARIES Celerity::celerity_runtime Celerity::ccto
  )

  add_sycl_to_target(
    TARGET ${SDK_ADD_CELERITY_TARGET}
    SOURCES ${SDK_ADD_CELERITY_SOURCES}
  )

  if(MSVC)
    # Copy ccto DLL to target build location on Windows
    add_custom_command(TARGET ${SDK_ADD_CELERITY_TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_CELERITY_INSTALL_PATH}/lib/OpenCL.dll"
      $<TARGET_FILE_DIR:${SDK_ADD_CELERITY_TARGET}>
    )
  endif()
endfunction()

