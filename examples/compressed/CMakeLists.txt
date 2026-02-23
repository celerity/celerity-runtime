cmake_minimum_required(VERSION 3.13)
project(compressed LANGUAGES CXX)

find_package(Celerity 0.7.0 REQUIRED)

function(add_compressed_variant target_name)
	add_executable(${target_name} compressed.cc)
	set_target_properties(
		${target_name}
		PROPERTIES
			CXX_STANDARD ${CELERITY_CXX_STANDARD}
			RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/compress"
			RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/compress"
			RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/compress"
			RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/compress"
			RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/compress")
	target_compile_definitions(${target_name} PRIVATE ${ARGN})
	add_celerity_to_target(TARGET ${target_name} SOURCES compressed.cc)
endfunction()

add_compressed_variant(compressed_point_cloud_element USE_ELEMENT_COMPRESSION USE_POINT_CLOUD_COMPRESSION)
add_compressed_variant(compressed_point_cloud_local USE_LOCAL_COMPRESSION USE_POINT_CLOUD_COMPRESSION)
add_compressed_variant(compressed_point_cloud_global USE_GLOBAL_COMPRESSION USE_POINT_CLOUD_COMPRESSION)
add_compressed_variant(compressed_point_cloud_with_dep_local USE_LOCAL_COMPRESSION USE_POINT_CLOUD_WITH_DEP_COMPRESSION)
add_compressed_variant(compressed_point_cloud_with_dep_global USE_GLOBAL_COMPRESSION USE_POINT_CLOUD_WITH_DEP_COMPRESSION)
add_compressed_variant(compressed_zcurve_local USE_LOCAL_COMPRESSION USE_ZCURVE_HYBRID)
add_compressed_variant(compressed_zcurve_global USE_GLOBAL_COMPRESSION USE_ZCURVE_HYBRID)
add_compressed_variant(compressed_uncompressed USE_UNCOMPRESSED)
