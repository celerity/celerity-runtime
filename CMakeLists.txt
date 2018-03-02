cmake_minimum_required(VERSION 3.8)
project(celerity_full_stack_example)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(FindComputeCpp)
find_package(Boost 1.66.0 REQUIRED)

add_executable(
	full_stack_example
	main.cc
	celerity_runtime.h
	celerity_runtime.cc
)

target_include_directories(
	full_stack_example
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/vendor
	${Boost_INCLUDE_DIRS}
	${COMPUTECPP_INCLUDE_DIRECTORY}
)

add_sycl_to_target(
	full_stack_example
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/main.cc
)

# TESTS

include(ucm)
add_subdirectory(googletest)

add_executable(
	runtime_tests
	runtime_tests.cc
	celerity_runtime.cc
)

find_package(OpenCL REQUIRED)

target_include_directories(
	runtime_tests
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/vendor
	${Boost_INCLUDE_DIRS}
	${OpenCL_INCLUDE_DIR}
	${COMPUTECPP_INCLUDE_DIRECTORY}
)

target_link_libraries(
	runtime_tests
	PUBLIC
	gtest_main
	# Link ComputeCpp libs manually instead of calling add_sycl_to_target()
	# since we don't want to include the kernel compilation step.
	$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}>
	$<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:${COMPUTECPP_RUNTIME_LIBRARY}>
	${OpenCL_LIBRARIES}
)

if (MSVC)
	# GoogleTest really wants to build against the static CRT, so we have to
	# configure the test executable accordingly. Unfortunately we cannot use
	# ucm_set_runtime(STATIC), because that would set the flags globally
	# which in turn breaks the ComputeCpp libs, which want dynamic CRT.
	ucm_gather_flags(0 flags_configs)
	foreach(flags ${flags_configs})
		if(${flags} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" new_flags "${${flags}}")
			string(REPLACE " " ";" new_flags ${new_flags})
			if(flags MATCHES "_DEBUG")
				target_compile_options(runtime_tests PUBLIC
					"$<$<CONFIG:DEBUG>:${new_flags}>")
			elseif(flags MATCHES "_RELEASE")
				target_compile_options(runtime_tests PUBLIC
					"$<$<CONFIG:RELEASE>:${new_flags}>")
			elseif(flags MATCHES "_MINSIZEREL")
				target_compile_options(runtime_tests PUBLIC
					"$<$<CONFIG:MINSIZEREL>:${new_flags}>")
			elseif(flags MATCHES "_RELWITHDEBINFO")
				target_compile_options(runtime_tests PUBLIC
					"$<$<CONFIG:RELWITHDEBINFO>:${new_flags}>")
			endif()
		endif()
	endforeach()
endif()
