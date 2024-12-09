# Celerity Legacy Branch: CCGrid / Multi-GPU

This is the version of Celerity used in benchmarking for the CCGrid 2023 paper
"An asynchronous dataflow-driven execution model for distributed accelerator computing".

It is based on the old buffer manager runtime, with experimental multi-GPU support and
CUDA specific hacks for asynchronous copies using CUDA streams. Reductions and many tests
are broken in this version.

Updates to dependencies and code interfacing with SYCL has been updated to allow building
with recent compilers and SYCL implementations (= AdaptiveCpp).
