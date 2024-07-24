cmake_minimum_required(VERSION 3.13)
project(syncing LANGUAGES CXX)

find_package(Celerity 0.5.0 REQUIRED)

add_executable(reduction reduction.cc)
set_property(TARGET reduction PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET reduction SOURCES reduction.cc)
