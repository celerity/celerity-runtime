add_executable(
  simple
  simple.cc
)

set_property(TARGET simple PROPERTY CXX_STANDARD 14)

target_include_directories(
  simple
  PUBLIC
  ${Celerity_INCLUDE_DIRS}
)

target_link_libraries(
  simple
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  simple
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/simple.cc
)
