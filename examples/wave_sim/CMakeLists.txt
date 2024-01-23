cmake_minimum_required(VERSION 3.13)
project(wave_sim LANGUAGES CXX)

find_package(Celerity 0.5.0 REQUIRED)

add_executable(wave_sim wave_sim.cc)
set_property(TARGET wave_sim PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET wave_sim SOURCES wave_sim.cc)
