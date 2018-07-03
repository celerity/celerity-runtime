add_executable(
  wave_sim
  wave_sim.cc
)

set_property(TARGET wave_sim PROPERTY CXX_STANDARD 14)

target_include_directories(
  wave_sim
  PUBLIC
  ${Celerity_INCLUDE_DIRS}
)

target_link_libraries(
  wave_sim
  PUBLIC
  celerity_runtime
)

add_sycl_to_target(
  wave_sim
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/wave_sim.cc
)
