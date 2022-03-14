if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0") 
    cmake_policy(SET CMP0110 NEW) # don't spam about add_test names..
endif()

include("${PROJECT_SOURCE_DIR}/vendor/Catch2/extras/ParseAndAddCatchTests.cmake")

# Function for setting all relevant test parameters
function(set_test_target_parameters TARGET SOURCE)
  target_link_libraries(${TARGET} PUBLIC Catch2::Catch2)
  set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${TARGET} PROPERTY FOLDER "tests")
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3 /bigobj)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wextra -Wno-unused-parameter -Wno-unused-variable)
  endif()

  add_celerity_to_target(TARGET ${TARGET} SOURCES ${SOURCE})
endfunction()

# Add includes to library so they show up in IDEs
file(GLOB_RECURSE TEST_INCLUDES *.h)

set(TEST_TARGETS
  accessor_tests
  buffer_manager_tests
  graph_generation_tests
  graph_gen_granularity_tests
  graph_gen_reduction_tests
  graph_gen_transfer_tests
  graph_compaction_tests
  intrusive_graph_tests
  region_map_tests
  runtime_tests
  runtime_deprecation_tests
  sycl_tests
  task_graph_tests
  device_selection_tests
)

add_library(test_main test_main.cc)
set_test_target_parameters(test_main test_main.cc)

set(TEST_OBJ_LIST "")
foreach(TEST_TARGET ${TEST_TARGETS})
  # Build test obj file
  set(TEST_SOURCE ${TEST_TARGET}.cc)
  set(TEST_OBJ ${TEST_TARGET}_OBJ)
  list(APPEND TEST_OBJ_LIST $<TARGET_OBJECTS:${TEST_OBJ}>)

  add_library(${TEST_OBJ} OBJECT ${TEST_SOURCE} ${TEST_INCLUDES})
  set_test_target_parameters(${TEST_OBJ} ${TEST_SOURCE})

  # Add test executable
  add_executable(${TEST_TARGET} $<TARGET_OBJECTS:${TEST_OBJ}>)
  target_link_libraries(${TEST_TARGET} PRIVATE test_main)
  set_test_target_parameters(${TEST_TARGET} "")
  
  # We use the (undocumented) per-file function as we otherwise run into
  # problems with ComputeCpp's generated integration headers.
  ParseAndAddCatchTests_ParseFile(${TEST_SOURCE} ${TEST_TARGET})
endforeach()

# Add all_tests executable
add_executable(all_tests ${TEST_OBJ_LIST})
target_link_libraries(all_tests PRIVATE test_main)
set_test_target_parameters(all_tests "")

# Unit benchmark executable
add_executable(benchmarks benchmarks.cc benchmark_reporters.cc)
target_link_libraries(benchmarks PRIVATE test_main)
set_test_target_parameters(benchmarks benchmarks.cc)

add_subdirectory(system)
