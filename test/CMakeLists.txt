include(Download_Catch2)
include(ParseAndAddCatchTests)

if(MSVC)
  file(GLOB_RECURSE TEST_INCLUDES *.h)
endif()

 # iterate over all cpp files in the current directory
file(GLOB_RECURSE ALL_TEST_CPP *.cc)
foreach(ONE_TEST_CPP ${ALL_TEST_CPP})

  # get file name without extension and use it as target name
  get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CPP} NAME_WE)
  set(TARGET_NAME ${ONE_TEST_EXEC})

  # add test executable
  add_executable(${TARGET_NAME} ${ONE_TEST_CPP} ${TEST_INCLUDES})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC})

  set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER "tests")

  target_include_directories(
    ${TARGET_NAME}
    PUBLIC
    ${Catch2_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC
    celerity_runtime
  )

  # Determine whether we want the device compiler to run on this test or not.
  # This can be enabled by adding "#define COMPILE_SYCL_KERNELS 1" to the first line of the file.
  file(READ ${ONE_TEST_CPP} FILE_CONTENTS LIMIT 32)
  if(${FILE_CONTENTS} MATCHES "^#define COMPILE_SYCL_KERNELS 1.*")
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${ONE_TEST_CPP})
  else()
    add_sycl_to_target(TARGET ${TARGET_NAME})
  endif()

  if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3 /bigobj)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wextra -Wno-unused-parameter)
  endif()

  ParseFile(${ONE_TEST_CPP} ${TARGET_NAME})

endforeach()
