add_executable(
  matmul_tiled
  matmul_tiled.cc
)

set_property(TARGET matmul_tiled PROPERTY CXX_STANDARD 17)

target_link_libraries(
  matmul_tiled
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET matmul_tiled
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/matmul_tiled.cc
)

if(MSVC)
  target_compile_options(matmul_tiled PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(matmul_tiled PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
