cmake_minimum_required(VERSION 3.13)
project(syncing LANGUAGES CXX)

find_package(Celerity 0.3.2 REQUIRED)

add_executable(syncing syncing.cc)
set_property(TARGET syncing PROPERTY CXX_STANDARD 17)
add_celerity_to_target(TARGET syncing SOURCES syncing.cc)
