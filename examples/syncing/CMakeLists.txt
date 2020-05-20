add_executable(
  syncing
  syncing.cc
)

set_property(TARGET syncing PROPERTY CXX_STANDARD 17)

target_link_libraries(
  syncing
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  TARGET syncing
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/syncing.cc
)

if(MSVC)
  target_compile_options(syncing PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(syncing PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
