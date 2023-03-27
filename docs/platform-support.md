---
id: platform-support
title: Officially Supported Platforms
sidebar_label: Platform Support
---

# Officially Supported Platforms

The most recent version of Celerity aims to support the following environments:

* hipSYCL ≥ revision [`24980221`](https://github.com/illuhad/hipSYCL/commit/24980221), with
  * Clang ≥ 10.0
  * CUDA ≥ 11.0
  * on NVIDIA hardware with compute capability ≥ 7.0
  * or on CPUs via OpenMP
* DPC++ ≥ revision [`61e51015`](https://github.com/intel/llvm/commit/61e51015)
  * on Intel hardware

ComputeCpp is currently not supported.

## Continuously Tested Configurations

We automatically verify Celerity's build process and test suites against a select number of system configurations.

Those are:

| SYCL       | SYCL version                                                                   | OS           | Build type     |
|------------|--------------------------------------------------------------------------------|--------------|----------------|
| DPC++      | [`61e51015`](https://github.com/intel/llvm/commit/61e51015)                    | Ubuntu 20.04 | Debug          |
| DPC++      | [`HEAD`](https://github.com/intel/llvm/)                                       | Ubuntu 22.04 | Debug, Release |
| hipSYCL    | [`24980221`](https://github.com/illuhad/hipSYCL/commit/24980221) (CUDA 11.0.3) | Ubuntu 20.04 | Debug          |
| hipSYCL    | [`HEAD`](https://github.com/illuhad/hipSYCL) (CUDA 12.1.0)                     | Ubuntu 22.04 | Debug, Release |
