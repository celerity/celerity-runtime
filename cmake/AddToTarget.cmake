if(CELERITY_SYCL_IMPL STREQUAL "DPC++")
  function(add_sycl_to_target)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args)
    cmake_parse_arguments(ADD_SYCL
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )
    list(PREPEND DPCPP_FLAGS
      -fsycl
      -sycl-std=2020
      "-fsycl-targets=${CELERITY_DPCPP_TARGETS}"
      -Wno-sycl-strict  # -Wsycl-strict produces false-positive warnings in DPC++'s own SYCL headers as of 2022-10-06
    )
    target_compile_options(${ADD_SYCL_TARGET} PUBLIC ${DPCPP_FLAGS})
    target_link_options(${ADD_SYCL_TARGET} PUBLIC ${DPCPP_FLAGS})
  endfunction()
elseif(CELERITY_SYCL_IMPL STREQUAL "SimSYCL")
  function(add_sycl_to_target)
    set(options)
    set(one_value_args TARGET)
    set(multi_value_args)
    cmake_parse_arguments(ADD_SYCL
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )
  endfunction()
endif()

function(add_celerity_to_target)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(ADD_CELERITY
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  set_property(
    TARGET ${ADD_CELERITY_TARGET}
    APPEND PROPERTY LINK_LIBRARIES ${CELERITY_RUNTIME_LIBRARY}
  )

  add_sycl_to_target(
    TARGET ${ADD_CELERITY_TARGET}
    SOURCES ${ADD_CELERITY_SOURCES}
  )
endfunction()
