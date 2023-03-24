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
* DPC++ ≥ revision [`3fd08509`](https://github.com/intel/llvm/commit/3fd08509)
  * on Intel hardware

ComputeCpp is currently not supported.

## Continuously Tested Configurations

We automatically verify Celerity's build process and test suites against a select number of system configurations.

Those are:

| SYCL       | SYCL version                                                                   | OS           | Build type     |
|------------|--------------------------------------------------------------------------------|--------------|----------------|
| DPC++      | [`3fd08509`](https://github.com/intel/llvm/commit/3fd08509)                    | Ubuntu 20.04 | Debug          |
| DPC++      | [`HEAD`](https://github.com/intel/llvm/)                                       | Ubuntu 22.04 | Debug, Release |
| hipSYCL    | [`7b00e2ef`](https://github.com/illuhad/hipSYCL/commit/7b00e2ef) (CUDA 11.0.3) | Ubuntu 20.04 | Debug          |
| hipSYCL    | [`HEAD`](https://github.com/illuhad/hipSYCL) (CUDA 11.7.0)                     | Ubuntu 22.04 | Debug, Release |
