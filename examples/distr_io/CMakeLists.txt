cmake_minimum_required(VERSION 3.13)
project(distr_io LANGUAGES CXX)

find_package(Celerity 0.5.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(HDF5 REQUIRED IMPORTED_TARGET hdf5-openmpi hdf5-1.12.0 hdf5)

add_executable(distr_io distr_io.cc)
set_property(TARGET distr_io PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET distr_io SOURCES distr_io.cc)
target_link_libraries(distr_io PUBLIC PkgConfig::HDF5)
