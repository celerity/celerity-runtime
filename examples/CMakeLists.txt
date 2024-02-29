option(CELERITY_EXAMPLES_REQUIRE_HDF5 "Fail on missing HDF5 dependency instead of skipping the distr_io example" OFF)

function(add_example NAME)
  add_executable(
    "${NAME}"
    "${NAME}/${NAME}.cc"
  )

  set_property(TARGET "${NAME}" PROPERTY CXX_STANDARD "${CELERITY_CXX_STANDARD}")
  set_property(TARGET "${NAME}" PROPERTY RUNTIME_OUTPUT_DIRECTORY "${NAME}")
  set_property(TARGET "${NAME}" PROPERTY FOLDER "examples")

  add_celerity_to_target(
    TARGET "${NAME}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/${NAME}.cc"
  )

  if(MSVC)
    target_compile_options("${NAME}" PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options("${NAME}" PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

add_example(convolution)
add_example(hello_world)
add_example(matmul)
add_example(syncing)
add_example(wave_sim)
add_example(reduction)

# NOCOMMIT do not remove HDF5 example
