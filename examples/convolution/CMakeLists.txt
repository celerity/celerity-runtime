add_executable(
  convolution
  convolution.cc
)

set_property(TARGET convolution PROPERTY CXX_STANDARD 17)

target_link_libraries(
  convolution
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET convolution
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cc
)

if(MSVC)
  target_compile_options(convolution PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(convolution PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
