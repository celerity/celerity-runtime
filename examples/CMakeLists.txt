option(CELERITY_EXAMPLES_REQUIRE_HDF5 "Fail on missing HDF5 dependency instead of skipping the distr_io example" OFF)

function(add_example NAME)
  add_executable(
    "${NAME}"
    "${NAME}/${NAME}.cc"
  )

  set_property(TARGET "${NAME}" PROPERTY CXX_STANDARD "${CELERITY_CXX_STANDARD}")
  set_property(TARGET "${NAME}" PROPERTY RUNTIME_OUTPUT_DIRECTORY "${NAME}")
  set_property(TARGET "${NAME}" PROPERTY FOLDER "examples")

  add_celerity_to_target(
    TARGET "${NAME}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/${NAME}.cc"
  )

  if(MSVC)
    target_compile_options("${NAME}" PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options("${NAME}" PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

add_example(convolution)
add_example(hello_world)
add_example(matmul)
add_example(syncing)
add_example(wave_sim)

if(CELERITY_FEATURE_SCALAR_REDUCTIONS)
  add_example(reduction)
endif()

find_package(PkgConfig QUIET)
set(HDF5_FOUND OFF)
if(PkgConfig_FOUND)
  pkg_search_module(HDF5 IMPORTED_TARGET hdf5-openmpi hdf5-1.12.0 hdf5)
  if(HDF5_FOUND)
    add_example(distr_io)
    target_link_libraries(distr_io PUBLIC PkgConfig::HDF5)
  endif()
endif()
if(CELERITY_EXAMPLES_REQUIRE_HDF5 AND NOT HDF5_FOUND)
  message(FATAL_ERROR "HDF5 not found, but required by CELERITY_EXAMPLES_REQUIRE_HDF5")
endif()
