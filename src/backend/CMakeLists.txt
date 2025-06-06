set(CUDA_OR_CUDA_TOOLKIT_FOUND FALSE)
if(NOT CELERITY_SYCL_IMPL STREQUAL "SimSYCL")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit)
    set(CUDA_OR_CUDA_TOOLKIT_FOUND ${CUDAToolkit_FOUND})
  else()
    find_package(CUDA)
    set(CUDA_OR_CUDA_TOOLKIT_FOUND ${CUDA_FOUND})
  endif()
  # find_package(LevelZero QUIET) # TODO: Need find module?
  # find_package(ROCM QUIET) # TODO: Need find module?

  option(CELERITY_ENABLE_CUDA_BACKEND "Enable optimized code paths for CUDA backends" ${CUDA_OR_CUDA_TOOLKIT_FOUND})
else()
  set(CELERITY_ENABLE_CUDA_BACKEND FALSE)
endif()

if(CELERITY_ENABLE_CUDA_BACKEND AND NOT CUDA_OR_CUDA_TOOLKIT_FOUND)
  # Run find_package again to emit error message
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit REQUIRED)
  else()
    find_package(CUDA REQUIRED)
  endif()
endif()

add_library(celerity_backends STATIC sycl_backend.cc sycl_generic_backend.cc)
# We link against the RT here to get all of its transitive properties (circular linking is allowed for static libraries).
target_link_libraries(celerity_backends PRIVATE celerity_runtime)
add_sycl_to_target(TARGET celerity_backends SOURCES)

if(CELERITY_ENABLE_CUDA_BACKEND)
  target_sources(celerity_backends PRIVATE sycl_cuda_backend.cc)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    target_link_libraries(celerity_backends PUBLIC CUDA::cudart)
  else()
    target_link_libraries(celerity_backends PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(celerity_backends PUBLIC ${CUDA_INCLUDE_DIRS})
  endif()
  target_compile_definitions(celerity_backends PUBLIC "CELERITY_DETAIL_BACKEND_CUDA_ENABLED=1")
  message(STATUS "CUDA backend enabled")
else()
  target_compile_definitions(celerity_backends PUBLIC "CELERITY_DETAIL_BACKEND_CUDA_ENABLED=0")
endif()
