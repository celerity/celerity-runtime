cmake_minimum_required(VERSION 3.13)
project(wave_sim_compression LANGUAGES CXX)

find_package(Celerity 0.7.0 REQUIRED)

add_executable(wave_sim_compression wave_sim_compression.cc)
add_celerity_to_target(TARGET wave_sim_compression SOURCES wave_sim_compression.cc)
