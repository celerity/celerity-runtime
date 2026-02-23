cmake_minimum_required(VERSION 3.13)
project(scratch_buffer LANGUAGES CXX)

find_package(Celerity 0.7.0 REQUIRED)

add_executable(combined_compression combined_compression.cc)
add_celerity_to_target(TARGET combined_compression SOURCES combined_compression.cc)
