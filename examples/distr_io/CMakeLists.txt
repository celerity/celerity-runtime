# This subdirectory is only compiled if HDF5 was found by pkg-config

add_executable(
  distr_io
  distr_io.cc
)

set_property(TARGET distr_io PROPERTY CXX_STANDARD 17)

target_link_libraries(
  distr_io
  PUBLIC
  celerity_runtime
  PkgConfig::HDF5
)

add_sycl_to_target(
  TARGET distr_io
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/distr_io.cc
)

target_compile_options(distr_io PRIVATE -Wall -Wextra -Wno-unused-parameter)
