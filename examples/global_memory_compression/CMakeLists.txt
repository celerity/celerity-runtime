cmake_minimum_required(VERSION 3.13)
project(scratch_buffer LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(global_memory_compression global_memory_compression.cc)
add_celerity_to_target(TARGET global_memory_compression SOURCES global_memory_compression.cc)
