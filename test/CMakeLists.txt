if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0") 
    cmake_policy(SET CMP0110 NEW) # don't spam about add_test names..
endif()

include("${PROJECT_SOURCE_DIR}/vendor/Catch2/contrib/ParseAndAddCatchTests.cmake")

# Add includes to library so they show up in IDEs
file(GLOB_RECURSE TEST_INCLUDES *.h)

set(TEST_TARGETS
  runtime_tests
  graph_generation_tests
  graph_compaction_tests
  intrusive_graph_tests
)

add_library(unit_test_suite unit_test_suite.cc unit_test_suite_celerity.cc)

target_link_libraries(
  unit_test_suite
  PUBLIC
  celerity_runtime
  Catch2::Catch2
)

set_property(TARGET unit_test_suite PROPERTY CXX_STANDARD 17)

add_sycl_to_target(TARGET unit_test_suite SOURCES unit_test_suite_celerity.cc)

foreach(TEST_TARGET ${TEST_TARGETS})

  set(TEST_SOURCE ${TEST_TARGET}.cc)

  # add test executable
  add_executable(${TEST_TARGET} ${TEST_SOURCE} ${TEST_INCLUDES})
  target_link_libraries(${TEST_TARGET} PRIVATE unit_test_suite)

  set_property(TARGET ${TEST_TARGET} PROPERTY CXX_STANDARD 17)
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
