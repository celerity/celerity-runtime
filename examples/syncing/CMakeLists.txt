cmake_minimum_required(VERSION 3.13)
project(syncing LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(syncing syncing.cc)
add_celerity_to_target(TARGET syncing SOURCES syncing.cc)
