cmake_minimum_required(VERSION 3.13)
project(matmul LANGUAGES CXX)

find_package(Celerity 0.3.2 REQUIRED)

add_executable(matmul matmul.cc)
set_property(TARGET matmul PROPERTY CXX_STANDARD 17)
add_celerity_to_target(TARGET matmul SOURCES matmul.cc)
