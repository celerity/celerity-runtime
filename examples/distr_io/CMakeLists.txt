cmake_minimum_required(VERSION 3.13)
project(distr_io LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)
if(NOT CELERITY_ENABLE_MPI)
  message(SEND_ERROR "Your Celerity installation is built without MPI support.\nSkip this example.")
endif()

find_package(PkgConfig REQUIRED)
pkg_search_module(HDF5 REQUIRED IMPORTED_TARGET hdf5-openmpi hdf5-1.12.0 hdf5)

add_executable(distr_io distr_io.cc)
add_celerity_to_target(TARGET distr_io SOURCES distr_io.cc)
target_link_libraries(distr_io PUBLIC PkgConfig::HDF5)
