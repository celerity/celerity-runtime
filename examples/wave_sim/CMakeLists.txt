cmake_minimum_required(VERSION 3.13)
project(wave_sim LANGUAGES CXX)

find_package(Celerity 0.3.2 REQUIRED)

add_executable(wave_sim wave_sim.cc)
set_property(TARGET wave_sim PROPERTY CXX_STANDARD 17)
add_celerity_to_target(TARGET wave_sim SOURCES wave_sim.cc)
