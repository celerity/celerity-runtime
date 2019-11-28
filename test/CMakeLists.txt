include(Download_Catch2)
include(ParseAndAddCatchTests)

if(MSVC)
  file(GLOB_RECURSE TEST_INCLUDES *.h)
endif()

set(TEST_TARGETS
  runtime_tests
  graph_generation_tests
)

add_library(unit_test_suite unit_test_suite.cc unit_test_suite_celerity.cc)

target_include_directories(
  unit_test_suite
  PUBLIC
  ${Catch2_INCLUDE_DIRS}
)

target_link_libraries(
  unit_test_suite
  PUBLIC
  celerity_runtime
)

set_property(TARGET unit_test_suite PROPERTY CXX_STANDARD 14)

add_sycl_to_target(TARGET unit_test_suite SOURCES unit_test_suite_celerity.cc)

foreach(TEST_TARGET ${TEST_TARGETS})

  set(TEST_SOURCE ${TEST_TARGET}.cc)

  # add test executable
  add_executable(${TEST_TARGET} ${TEST_SOURCE} ${TEST_INCLUDES})
  target_link_libraries(${TEST_TARGET} PRIVATE unit_test_suite)

  set_property(TARGET ${TEST_TARGET} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${TEST_TARGET} PROPERTY FOLDER "tests")

  add_sycl_to_target(TARGET ${TEST_TARGET} SOURCES ${TEST_SOURCE})

  if(MSVC)
    target_compile_options(${TEST_TARGET} PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3 /bigobj)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TEST_TARGET} PRIVATE -Wall -Wextra -Wextra -Wno-unused-parameter)
  endif()

  # We use the (undocumented) per-file function as we otherwise run into
  # problems with ComputeCpp's generated integration headers.
  ParseAndAddCatchTests_ParseFile(${TEST_SOURCE} ${TEST_TARGET})

endforeach()
