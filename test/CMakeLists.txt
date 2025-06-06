if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0")
    cmake_policy(SET CMP0110 NEW) # don't spam about add_test names..
endif()

include(ParseAndAddCatchTests)

# Function for setting all relevant test parameters
function(set_test_target_parameters TARGET SOURCE)
  target_include_directories(${TARGET} PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_link_libraries(${TARGET} PUBLIC Catch2::Catch2)
  set_property(TARGET ${TARGET} PROPERTY FOLDER "tests")
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3 /bigobj)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wextra -Wno-unused-parameter -Wno-unused-variable)
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET} PRIVATE -Wno-array-bounds -Wno-stringop-overflow)  # false positives with GCC 13.2
  endif()

  add_celerity_to_target(TARGET ${TARGET} SOURCES ${SOURCE})
endfunction()

# Add includes to library so they show up in IDEs
file(GLOB_RECURSE TEST_INCLUDES *.h)

set(TEST_TARGETS
  accessor_tests
  affinity_tests
  backend_tests
  command_graph_general_tests
  command_graph_granularity_tests
  command_graph_reduction_tests
  command_graph_transfer_tests
  debug_naming_tests
  double_buffered_queue_tests
  executor_tests
  graph_compaction_tests
  grid_tests
  handler_tests
  hint_tests
  intrusive_graph_tests
  instruction_graph_grid_tests
  instruction_graph_memory_tests
  instruction_graph_misc_tests
  instruction_graph_p2p_tests
  instruction_graph_reduction_tests
  named_threads_tests
  nd_memory_tests
  out_of_order_engine_tests
  print_graph_tests
  print_utils_tests
  region_map_tests
  range_tests
  range_mapper_tests
  receive_arbiter_tests
  runtime_tests
  runtime_deprecation_tests
  scheduler_tests
  sycl_tests
  split_tests
  task_graph_tests
  test_utils_tests
  thread_queue_tests
  utils_tests
  device_selection_tests
)

add_library(test_main test_main.cc test_utils.cc grid_test_utils.cc)
set_test_target_parameters(test_main test_main.cc test_utils.cc grid_test_utils.cc)

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

# Add instruction_graph_tests executable
add_executable(instruction_graph_tests
  $<TARGET_OBJECTS:instruction_graph_grid_tests_OBJ>
  $<TARGET_OBJECTS:instruction_graph_memory_tests_OBJ>
  $<TARGET_OBJECTS:instruction_graph_misc_tests_OBJ>
  $<TARGET_OBJECTS:instruction_graph_p2p_tests_OBJ>
  $<TARGET_OBJECTS:instruction_graph_reduction_tests_OBJ>)
target_link_libraries(instruction_graph_tests PRIVATE test_main)
set_test_target_parameters(instruction_graph_tests "")

# Add all_tests executable
add_executable(all_tests ${TEST_OBJ_LIST})
target_link_libraries(all_tests PRIVATE test_main)
set_test_target_parameters(all_tests "")

# Unit benchmark executable
add_executable(benchmarks dag_benchmarks.cc grid_benchmarks.cc system_benchmarks.cc benchmark_reporters.cc)
target_link_libraries(benchmarks PRIVATE test_main)
set_test_target_parameters(benchmarks dag_benchmarks.cc grid_benchmarks.cc system_benchmarks.cc)

if(CELERITY_ENABLE_MPI)
  add_subdirectory(system)
endif()

if(CELERITY_DETAIL_INTEGRATION_TESTING)
  add_subdirectory(integration)
endif()

# Optionally link region map tests against Cairo for debug drawing

find_path(CAIRO_INCLUDE_DIRS
    NAMES cairo.h
    PATH_SUFFIXES cairo
)

find_library(CAIRO_LIBRARIES
    NAMES cairo
)

if(CAIRO_INCLUDE_DIRS AND CAIRO_LIBRARIES)
  message(STATUS "Building tests with cairo support: ${CAIRO_LIBRARIES}")
  foreach(TEST_TARGET region_map_tests;region_map_tests_OBJ;test_main)
    target_compile_definitions(${TEST_TARGET} PRIVATE CELERITY_DETAIL_HAVE_CAIRO=1)
    target_include_directories(${TEST_TARGET} PRIVATE ${CAIRO_INCLUDE_DIRS})
    target_link_libraries(${TEST_TARGET} PRIVATE ${CAIRO_LIBRARIES})
  endforeach()
  target_link_libraries(all_tests PRIVATE ${CAIRO_LIBRARIES})
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  add_subdirectory(debug)
endif()
