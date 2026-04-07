option(CELERITY_EXAMPLES_REQUIRE_HDF5 "Fail on missing HDF5 dependency instead of skipping the distr_io example" OFF)

function(add_example NAME)
  add_executable(
    "${NAME}"
    "${NAME}/${NAME}.cc"
  )

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

function(add_example_variant NAME)
  add_executable(
    "${NAME}"
    "compressed/compressed.cc"
  )

  set_property(TARGET "${NAME}" PROPERTY RUNTIME_OUTPUT_DIRECTORY "${NAME}")
  set_property(TARGET "${NAME}" PROPERTY FOLDER "examples")

  add_celerity_to_target(
    TARGET "${NAME}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/compressed/compressed.cc"
  )

  target_compile_definitions("${NAME}" PRIVATE ${ARGN})

  if(MSVC)
    target_compile_options("${NAME}" PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options("${NAME}" PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

function(add_compressed_variant NAME)
  add_executable(
    "${NAME}"
    "compressed/compressed.cc"
  )

  set_target_properties(
    "${NAME}"
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/compress"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/compress"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/compress"
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/compress"
      RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/compress"
      FOLDER "examples"
  )

  add_celerity_to_target(
    TARGET "${NAME}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/compressed/compressed.cc"
  )

  target_compile_definitions("${NAME}" PRIVATE ${ARGN})

  if(MSVC)
    target_compile_options("${NAME}" PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options("${NAME}" PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

function(add_wave_sim_compression_variant NAME DEFINE)
  add_executable(
    "${NAME}"
    "wave_sim_compression/wave_sim_compression.cc"
  )

  set_property(TARGET "${NAME}" PROPERTY RUNTIME_OUTPUT_DIRECTORY "wave_sim_compression")
  set_property(TARGET "${NAME}" PROPERTY FOLDER "examples")

  add_celerity_to_target(
    TARGET "${NAME}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/wave_sim_compression/wave_sim_compression.cc"
  )

  target_compile_definitions("${NAME}" PRIVATE "${DEFINE}")

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
add_example(reduction)
add_example(nd_range_wave_sim)
add_compressed_variant(compressed_point_cloud_element USE_ELEMENT_COMPRESSION USE_POINT_CLOUD_COMPRESSION)
add_compressed_variant(compressed_point_cloud_local USE_LOCAL_COMPRESSION USE_POINT_CLOUD_COMPRESSION)
add_compressed_variant(compressed_point_cloud_global USE_GLOBAL_COMPRESSION USE_POINT_CLOUD_COMPRESSION)
add_compressed_variant(compressed_point_cloud_with_dep_local USE_LOCAL_COMPRESSION USE_POINT_CLOUD_WITH_DEP_COMPRESSION)
add_compressed_variant(compressed_point_cloud_with_dep_global USE_GLOBAL_COMPRESSION USE_POINT_CLOUD_WITH_DEP_COMPRESSION)
add_compressed_variant(compressed_zcurve_local USE_LOCAL_COMPRESSION USE_ZCURVE_HYBRID)
add_compressed_variant(compressed_zcurve_global USE_GLOBAL_COMPRESSION USE_ZCURVE_HYBRID)
add_compressed_variant(compressed_uncompressed USE_UNCOMPRESSED)
add_wave_sim_compression_variant(wave_sim_compression_element USE_ELEMENT_COMPRESSION)
add_wave_sim_compression_variant(wave_sim_compression_local USE_LOCAL_COMPRESSION)
add_wave_sim_compression_variant(wave_sim_compression_global USE_GLOBAL_COMPRESSION)

if(CELERITY_ENABLE_MPI)
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
endif()
