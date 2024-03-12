set(SYSTEM_TEST_TARGETS
  distr_tests
  mpi_tests
)

foreach(TEST_TARGET ${SYSTEM_TEST_TARGETS})
  set(TEST_SOURCE ${TEST_TARGET}.cc)

  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} PRIVATE test_main)
  set_test_target_parameters(${TEST_TARGET} ${TEST_SOURCE})

  ParseAndAddCatchTests_ParseFile(${TEST_SOURCE} ${TEST_TARGET})
endforeach()
