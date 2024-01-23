if(NOT CELERITY_DETAIL_INTEGRATION_TESTING)
  message(FATAL "Integration tests should not be compiled directly; use Python wrapper script instead")
endif()

function(add_integration_test TARGET SOURCE)
  add_executable(${TARGET} ${SOURCE})
  set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})

  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3 /bigobj)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wextra -Wno-unused-parameter)
  endif()

  add_celerity_to_target(TARGET ${TARGET} SOURCES ${SOURCE})
endfunction()

add_integration_test(backend backend.cc)