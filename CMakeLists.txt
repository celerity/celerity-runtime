cmake_minimum_required(VERSION 3.8)
project(celerity_full_stack_example)

find_package(Boost 1.66.0 REQUIRED)

add_executable(full_stack_example main.cc celerity_runtime.h celerity_runtime.cc)
set_property(TARGET full_stack_example PROPERTY CXX_STANDARD 17)
target_include_directories(full_stack_example PUBLIC ${Boost_INCLUDE_DIRS})
#target_link_libraries(full_stack_example Boost::)

