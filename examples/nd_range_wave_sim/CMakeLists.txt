cmake_minimum_required(VERSION 3.13)
project(nd_range_wave_sim LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(nd_range_wave_sim nd_range_wave_sim.cc)
add_celerity_to_target(TARGET nd_range_wave_sim SOURCES nd_range_wave_sim.cc)
