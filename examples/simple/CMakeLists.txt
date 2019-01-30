add_executable(
  simple
  simple.cc
)

set_property(TARGET simple PROPERTY CXX_STANDARD 14)

target_link_libraries(
  simple
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET simple
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/simple.cc
)

if(MSVC)
  target_compile_options(simple PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(simple PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
