cmake_minimum_required(VERSION 3.13)

if(DEFINED PROJECT_NAME)
  set(INCLUDED_AS_SUBPROJECT ON)
else()
  set(INCLUDED_AS_SUBPROJECT OFF)
endif()

file(STRINGS "../VERSION" Celerity_VERSION)
project(celerity_benchmarks VERSION "${Celerity_VERSION}" LANGUAGES CXX)

if(NOT INCLUDED_AS_SUBPROJECT)
  find_package(Celerity "${Celerity_VERSION}")
endif()

function(add_benchmark NAME)
  add_executable(
    "${NAME}"
    "${NAME}.cc"
  )

  set_property(TARGET "${NAME}" PROPERTY CXX_STANDARD 17)
  set_property(TARGET "${NAME}" PROPERTY FOLDER "benchmarks")

  add_celerity_to_target(
    TARGET "${NAME}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cc"
  )

  target_link_libraries("${NAME}" PUBLIC args)

  if(MSVC)
    target_compile_options("${NAME}" PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options("${NAME}" PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

add_benchmark(task_microbenchmark)
