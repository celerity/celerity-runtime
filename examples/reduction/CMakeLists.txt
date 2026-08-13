cmake_minimum_required(VERSION 3.13)
project(reduction LANGUAGES CXX)

find_package(Celerity 0.7.1 REQUIRED)

add_executable(reduction reduction.cc)
add_celerity_to_target(TARGET reduction SOURCES reduction.cc)
