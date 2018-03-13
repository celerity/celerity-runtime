# platform-spefic
#
# Windows: use gtest installed via vcpkg
# Linux: clone repo and build manually
#
include(ConfigCatch2)
include(ParseAndAddCatchTests)

 # iterate over all cpp files in the current directory
file(GLOB_RECURSE ALL_TEST_CPP *.cc)
foreach(ONE_TEST_CPP ${ALL_TEST_CPP})

  # get file name without extension and use it as target name
  get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CPP} NAME_WE)
  set(TARGET_NAME ${ONE_TEST_EXEC})

  # add test executable
  add_executable(${TARGET_NAME} ${ONE_TEST_CPP})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC})

  set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

  target_include_directories(
    ${TARGET_NAME}
    PUBLIC
    ${Celerity_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC
    celerity_runtime
  )

  add_sycl_to_target(
    ${TARGET_NAME}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ONE_TEST_CPP}
  )

  if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    copy_dlls(${TARGET_NAME})
  endif()

  ParseFile(${ONE_TEST_CPP} ${TARGET_NAME})
  #add_test(NAME ${TARGET_NAME} COMMAND ${ONE_TEST_EXEC})

endforeach()

