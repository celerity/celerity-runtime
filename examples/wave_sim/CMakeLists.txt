add_executable(
  wave_sim
  wave_sim.cc
)

set_property(TARGET wave_sim PROPERTY CXX_STANDARD 14)

target_link_libraries(
  wave_sim
  PUBLIC
  celerity_runtime
  # FIXME: This is a workaround for add_sycl_to_target linking OpenCL, which breaks clin on Linux
  # (Transitive dependencies like ccto via celerity_runtime seem to be added to the executable last)
  ccto
)

add_sycl_to_target(
  TARGET wave_sim
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/wave_sim.cc
)

if(MSVC)
  target_compile_options(wave_sim PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  # Copy ccto DLL
  add_custom_command(TARGET wave_sim
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:ccto>
    $<TARGET_FILE_DIR:wave_sim>
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(wave_sim PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
