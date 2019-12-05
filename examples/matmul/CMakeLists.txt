add_executable(
  matmul
  matmul.cc
)

add_executable(
  matmulChain
  matmulChain.cc
)

set_property(TARGET matmul PROPERTY CXX_STANDARD 14)

target_link_libraries(
  matmul
  PUBLIC
  celerity_runtime
)

target_link_libraries(
  matmulChain
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET matmul
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cc
)

add_sycl_to_target(
  TARGET matmulChain
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/matmulChain.cc
)

if(MSVC)
  target_compile_options(matmul PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  target_compile_options(matmulChain PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(matmul PRIVATE -Wall -Wextra -Wno-unused-parameter)
  target_compile_options(matmulChain PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
