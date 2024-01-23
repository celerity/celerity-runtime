cmake_minimum_required(VERSION 3.13)
project(hello_world LANGUAGES CXX)

find_package(Celerity 0.5.0 REQUIRED)

add_executable(hello_world hello_world.cc)
set_property(TARGET hello_world PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET hello_world SOURCES hello_world.cc)
