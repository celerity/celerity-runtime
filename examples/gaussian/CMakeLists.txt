add_executable(
  gaussian
  gaussian.cc
)

set_property(TARGET gaussian PROPERTY CXX_STANDARD 14)

target_link_libraries(
  gaussian
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET gaussian
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gaussian.cc
)

if(MSVC)
  target_compile_options(gaussian PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(gaussian PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
