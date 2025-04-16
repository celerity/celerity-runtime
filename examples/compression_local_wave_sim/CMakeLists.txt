cmake_minimum_required(VERSION 3.13)
project(compression_local_wave_sim LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(compression_local_wave_sim compression_local_wave_sim.cc)
add_celerity_to_target(TARGET compression_local_wave_sim SOURCES compression_local_wave_sim.cc)
