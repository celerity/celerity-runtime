add_executable(distr_tests distr_tests.cc)

target_link_libraries(distr_tests PRIVATE test_main)

set_property(TARGET distr_tests PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
set_property(TARGET distr_tests PROPERTY FOLDER "tests/system")

add_celerity_to_target(TARGET distr_tests SOURCES distr_tests.cc)

if(MSVC)
  target_compile_options(distr_tests PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3 /bigobj)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(distr_tests PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
endif()
