add_executable(
  simple
  simple.cc
)

set_property(TARGET simple PROPERTY CXX_STANDARD 14)

target_link_libraries(
  simple
  PUBLIC
  celerity_runtime
  # FIXME: This is a workaround for add_sycl_to_target linking OpenCL, which breaks clin on Linux
  # (Transitive dependencies like ccto via celerity_runtime seem to be added to the executable last)
  ccto
)

add_sycl_to_target(
  simple
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/simple.cc
)

if(MSVC)
  target_compile_options(simple PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  # Copy ccto DLL
  add_custom_command(TARGET simple
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:ccto>
    $<TARGET_FILE_DIR:simple>
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(simple PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
