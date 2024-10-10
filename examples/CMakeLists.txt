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
    target_compile_options("${NAME}" PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
  endif()
endfunction()

add_example(convolution)
add_example(hello_world)
add_example(matmul)
add_example(syncing)
add_example(wave_sim)
add_example(reduction)

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

add_example(umuguc)
add_executable(tilebuffer_tests umuguc/tilebuffer_tests.cc)
add_celerity_to_target(TARGET tilebuffer_tests SOURCES umuguc/tilebuffer_tests.cc)
target_link_libraries(tilebuffer_tests PUBLIC Catch2::Catch2WithMain)
target_include_directories(tilebuffer_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_property(TARGET tilebuffer_tests PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})

# Gmsh CMake integration is pretty rough...
set(ENABLE_BUILD_LIB ON)
fetch_content_from_submodule(gmsh vendor/gmsh)
add_example(fvm)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(SEND_ERROR "Need CMAKE_CXX_COMPILER=clang++ for OpenMP compatibility")
endif()
find_package(OpenMP REQUIRED)
find_library(gmp gmp REQUIRED)
target_link_libraries(fvm PRIVATE OpenMP::OpenMP_CXX gmp)
# NOTE: FOR SOME STUPID REASON, ON FIRST COMPILE "lib" IS NOT FOUND.
# Solution: Comment the next line out, reconfigure, comment in again, reconfigure - it works ¯\_(ツ)_/¯
target_link_libraries(fvm PRIVATE lib) # gmsh
target_include_directories(fvm PRIVATE ${CMAKE_SOURCE_DIR}/vendor/gmsh/api)

# enable asan for fvm
# target_compile_options(fvm PRIVATE -fsanitize=address)
# target_link_options(fvm PRIVATE -fsanitize=address)
