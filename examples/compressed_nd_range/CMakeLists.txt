cmake_minimum_required(VERSION 3.13)
project(compressed_nd_range LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(compressed_nd_range compressed_nd_range.cc)
set_property(TARGET compressed_nd_range PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET compressed_nd_range SOURCES compressed_nd_range.cc)
