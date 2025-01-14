cmake_minimum_required(VERSION 3.13)
project(convolution LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(convolution convolution.cc)
add_celerity_to_target(TARGET convolution SOURCES convolution.cc)
