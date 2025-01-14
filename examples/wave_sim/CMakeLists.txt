cmake_minimum_required(VERSION 3.13)
project(wave_sim LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(wave_sim wave_sim.cc)
add_celerity_to_target(TARGET wave_sim SOURCES wave_sim.cc)
