cmake_minimum_required(VERSION 3.13)
project(compressed LANGUAGES CXX)

find_package(Celerity 0.4.1 REQUIRED)

add_executable(compressed compressed.cc)
set_property(TARGET compressed PROPERTY CXX_STANDARD 17)
add_celerity_to_target(TARGET compressed SOURCES compressed.cc)