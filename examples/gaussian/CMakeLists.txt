add_executable(
  gaussian
  gaussian.cc
)

set_property(TARGET gaussian PROPERTY CXX_STANDARD 14)

target_include_directories(
  gaussian
  PUBLIC
  ${Celerity_INCLUDE_DIRS}
)

target_link_libraries(
  gaussian
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  gaussian
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/gaussian.cc
)
