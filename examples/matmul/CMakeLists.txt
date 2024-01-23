cmake_minimum_required(VERSION 3.13)
project(matmul LANGUAGES CXX)

find_package(Celerity 0.5.0 REQUIRED)

add_executable(matmul matmul.cc)
set_property(TARGET matmul PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(TARGET matmul SOURCES matmul.cc)
