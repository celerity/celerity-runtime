find_package(CUDAToolkit QUIET)
# find_package(LevelZero QUIET) # TODO: Need find module?
# find_package(ROCM QUIET) # TODO: Need find module?

# TODO: Make conditional on CELERITY_SYCL_IMPL?
option(CELERITY_ENABLE_CUDA_BACKEND "Enable optimized code paths for CUDA backends" ${CUDAToolkit_FOUND})
if(CELERITY_ENABLE_CUDA_BACKEND AND NOT CUDAToolkit_FOUND)
  # Run find_package again to emit error message
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(celerity_backends STATIC backend.cc generic_backend.cc)
set_property(TARGET celerity_backends PROPERTY CXX_STANDARD 17)
# We link against the RT here to get all of its transitive properties (circular linking is allowed for static libraries).
target_link_libraries(celerity_backends PRIVATE celerity_runtime)
add_sycl_to_target(TARGET celerity_backends SOURCES)

if(CELERITY_ENABLE_CUDA_BACKEND)
  target_sources(celerity_backends PRIVATE cuda_backend.cc)
  target_link_libraries(celerity_backends PUBLIC CUDA::cudart)
  target_compile_definitions(celerity_backends PUBLIC "CELERITY_DETAIL_BACKEND_CUDA_ENABLED=1")
  message(STATUS "CUDA backend enabled")
endif()
