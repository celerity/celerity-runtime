cmake_minimum_required(VERSION 3.13)
project(compressed_actually LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(compressed_actually compressed_actually.cc)
set_property(TARGET compressed_actually PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET compressed_actually SOURCES compressed_actually.cc)
