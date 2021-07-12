add_executable(
  reduction
  reduction.cc
)

set_property(TARGET reduction PROPERTY CXX_STANDARD 17)

target_link_libraries(
  reduction
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET reduction
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/reduction.cc
)

if(MSVC)
  target_compile_options(reduction PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(reduction PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
