function(test_cxx_compiler_is_dpcpp VAR)
  try_compile("${VAR}" "${PROJECT_BINARY_DIR}/Celerity/detect_dpcpp"
    SOURCES "${CELERITY_CMAKE_DIR}/sycl_test.cpp"
    COMPILE_DEFINITIONS -fsycl -sycl-std=2020
    )
endfunction()
