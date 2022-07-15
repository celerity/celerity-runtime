---
id: platform-support
title: Officially Supported Platforms
sidebar_label: Platform Support
---

# Officially Supported Platforms

The most recent version of Celerity aims to support the following environments:

* hipSYCL ≥ revision [`7b00e2ef`](https://github.com/illuhad/hipSYCL/commit/7b00e2ef), with
  * Clang ≥ 10.0
  * CUDA ≥ 11.0
  * on NVIDIA hardware with compute capability ≥ 7.0
  * or on CPUs via OpenMP
* ComputeCpp ≥ version 2.6.0
  * on Intel hardware
  * with [stable](https://developer.codeplay.com/products/computecpp/ce/download) and [experimental](https://developer.codeplay.com/products/computecpp/ce/download?experimental=true) compilers
* DPC++ ≥ revision [`7735139b`](https://github.com/intel/llvm/commit/7735139b)
  * on Intel hardware

## Continuously Tested Configurations

We automatically verify Celerity's build process and test suites against a select number of system configurations.

Those are:

| SYCL       | SYCL version                                                                   | OS           | Build type     |
|------------|--------------------------------------------------------------------------------|--------------|----------------|
| ComputeCpp | 2.6.0, 2.7.0, 2.8.0 (stable)                                                   | Ubuntu 20.04 | Debug          |
| ComputeCpp | 2.9.0 (stable)                                                                 | Ubuntu 22.04 | Debug, Release |
| ComputeCpp | 2.9.0 (experimental compiler)                                                  | Ubuntu 22.04 | Debug, Release |
| DPC++      | [`7735139b`](https://github.com/intel/llvm/commit/7735139b)                    | Ubuntu 20.04 | Debug          |
| DPC++      | [`HEAD`](https://github.com/intel/llvm/)                                       | Ubuntu 22.04 | Debug, Release |
| hipSYCL    | [`7b00e2ef`](https://github.com/illuhad/hipSYCL/commit/7b00e2ef) (CUDA 11.0.3) | Ubuntu 20.04 | Debug          |
| hipSYCL    | [`HEAD`](https://github.com/illuhad/hipSYCL) (CUDA 11.7.0)                     | Ubuntu 22.04 | Debug, Release |
