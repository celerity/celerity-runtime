add_executable(
  matmul
  matmul.cc
)

set_property(TARGET matmul PROPERTY CXX_STANDARD 14)

target_include_directories(
  matmul
  PUBLIC
  ${Celerity_INCLUDE_DIRS}
)

target_link_libraries(
  matmul
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  matmul
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cc
)
