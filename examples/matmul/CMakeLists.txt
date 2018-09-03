add_executable(
  matmul
  matmul.cc
)

set_property(TARGET matmul PROPERTY CXX_STANDARD 14)

target_link_libraries(
  matmul
  PUBLIC
  celerity_runtime
  # FIXME: This is a workaround for add_sycl_to_target linking OpenCL, which breaks clin on Linux
  # (Transitive dependencies like ccto via celerity_runtime seem to be added to the executable last)
  ccto
)

add_sycl_to_target(
  matmul
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cc
)

if(MSVC)
  target_compile_options(matmul PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  # Copy ccto DLL
  add_custom_command(TARGET matmul
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:ccto>
    $<TARGET_FILE_DIR:matmul>
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(matmul PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
