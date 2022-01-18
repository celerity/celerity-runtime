# Split version string into components
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ ${CELERITY_VERSION})
set(CELERITY_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CELERITY_VERSION_MINOR ${CMAKE_MATCH_2})
set(CELERITY_VERSION_PATCH ${CMAKE_MATCH_3})

message(VERBOSE "Celerity version is ${CELERITY_VERSION_MAJOR}.${CELERITY_VERSION_MINOR}.${CELERITY_VERSION_PATCH}")

# Attempt to obtain git revision / dirty status
set(CELERITY_VERSION_GIT_REVISION "unknown")
set(CELERITY_VERSION_GIT_IS_DIRTY 0)

find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${CELERITY_SOURCE_DIR}"
    RESULT_VARIABLE EXIT_CODE
    OUTPUT_VARIABLE GIT_REVISION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(EXIT_CODE EQUAL 0)
    set(CELERITY_VERSION_GIT_REVISION ${GIT_REVISION})
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD
      WORKING_DIRECTORY "${CELERITY_SOURCE_DIR}"
      RESULT_VARIABLE CELERITY_VERSION_GIT_IS_DIRTY
      ERROR_QUIET
    )
  endif()
endif()

# Simply write resulting header (which is git-ignored) into include directory.
# This way we don't need to do any additional work
# for setting up include paths, during installation etc.
configure_file("${CMAKE_CURRENT_LIST_DIR}/version.h.in"
  "${CELERITY_SOURCE_DIR}/include/version.h"
  @ONLY
)
