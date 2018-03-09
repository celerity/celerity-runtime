cmake_minimum_required(VERSION 3.8)
project(celerity_full_stack_example)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/modules ${CMAKE_MODULE_PATH})

# Windows-specific:
#
# install libraries with vcpkg using the x64-windows triplet e.g.:
#
#	.\vcpkg.exe install boost-graph:x64-windows
#
# provide CMake toolchain file through -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg_toolchain_file>
# 
# the vcpkg tool chain file can be found in:
#
#	<vcpkg_root>/scripts/buildsystems/vcpkg.cmake
#
# either set it as command line parameter when invoking CMake 
#
#	OR
# 
# set it in the CMakeSettings.json when using the Visual Studio CMake integration
# 
# variables: [
#	{ 
#		name: "CMAKE_TOOLCHAIN_FILE", 
#		value: "<path_to_vcpkg_toolchain_file>"
#	}
# ]
#
#
# similarily the ComputeCpp root directory has to be set in order for CMake to find it -
# again, use one of the approaches above to set COMPUTECPP_PACKAGE_ROOT_DIR

find_package(Boost 1.66.0 COMPONENTS graph REQUIRED)
find_package(ComputeCpp REQUIRED)
find_package(OpenCL REQUIRED)

# Celerity runtime library

add_library(celerity_runtime celerity_runtime.cc)

set_property(TARGET celerity_runtime PROPERTY CXX_STANDARD 17)

target_include_directories(
	celerity_runtime
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/vendor
	${Boost_INCLUDE_DIRS}
	${OpenCL_INCLUDE_DIR}
	${ComputeCpp_INCLUDE_DIRECTORY}
)

target_link_libraries(
	celerity_runtime
	PUBLIC
	${Boost_LIBRARIES}
	${OpenCL_LIBRARIES}
)

add_sycl_to_target(
	celerity_runtime
	${CMAKE_CURRENT_BINARY_DIR}
	""
)

set(Celerity_LIBRARIES 
	celerity_runtime 
	${Boost_LIBRARIES}
	${OpenCL_LIBRARIES})

set(Celerity_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/vendor
	${Boost_INCLUDE_DIRS}
	${OpenCL_INCLUDE_DIR}
	${ComputeCpp_INCLUDE_DIRECTORY})

# full stack example

add_executable(
	full_stack_example
	main.cc
)

set_property(TARGET full_stack_example PROPERTY CXX_STANDARD 17)

target_include_directories(
	full_stack_example
	PUBLIC
	${Celerity_INCLUDE_DIRS}
)

target_link_libraries(
	full_stack_example
	PUBLIC
	${Celerity_LIBRARIES}
)

add_sycl_to_target(
	full_stack_example
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/main.cc
)

# Windows-specific:
#
# add post build step to copy ComputeCpp DLLs to the output directory
#
function(copy_dlls target)

	if (${CMAKE_BUILD_TYPE} MATCHES "Debug")

		get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_DEBUG_WE 
			${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} NAME_WE)

		get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY
			${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} DIRECTORY )

		add_custom_command(TARGET ${target}
		POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different	
			"${COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY}/${COMPUTECPP_RUNTIME_LIBRARY_DEBUG_WE}.dll"   
			$<TARGET_FILE_DIR:${target}>)

	else()

		get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_WE 
			${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} NAME_WE)

		get_filename_component(COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY
			${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} DIRECTORY )

		add_custom_command(TARGET ${target} 
		POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different	
			"${COMPUTECPP_RUNTIME_LIBRARY_DIRECTORY}/${COMPUTECPP_RUNTIME_LIBRARY_WE}.dll"    
			$<TARGET_FILE_DIR:${target}>)

	endif()

endfunction(copy_dlls)


if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	copy_dlls(full_stack_example)
endif()



