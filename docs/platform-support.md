---
id: platform-support
title: Officially Supported Platforms
sidebar_label: Platform Support
---

# Officially Supported Platforms

The most recent version of Celerity aims to support the following environments:

* hipSYCL ≥ revision [`24980221`](https://github.com/illuhad/hipSYCL/commit/24980221), with
  * CUDA ≥ 11.0
  * Clang ≥ 10.0 for CUDA &lt; 12.0, Clang ≥ 16.0 for CUDA ≥ 12.0
  * on NVIDIA hardware with compute capability ≥ 7.0
  * or on CPUs via OpenMP
* DPC++ ≥ revision [`61e51015`](https://github.com/intel/llvm/commit/61e51015)
  * on Intel hardware

ComputeCpp is currently not supported.

## Continuously Tested Configurations

We automatically verify Celerity's build process and test suites against a select number of system configurations.

Those are:

| SYCL       | SYCL version                                                                             | OS           | Build type     |
|------------|------------------------------------------------------------------------------------------|--------------|----------------|
| DPC++      | [`61e51015`](https://github.com/intel/llvm/commit/61e51015)                              | Ubuntu 20.04 | Debug          |
| DPC++      | [`HEAD`](https://github.com/intel/llvm/)                                                 | Ubuntu 22.04 | Debug, Release |
| hipSYCL    | [`24980221`](https://github.com/illuhad/hipSYCL/commit/24980221) (Clang 10, CUDA 11.0.3) | Ubuntu 20.04 | Debug          |
| hipSYCL    | [`24980221`](https://github.com/illuhad/hipSYCL/commit/24980221) (Clang 14, CUDA 11.8.0) | Ubuntu 22.04 | Debug, Release |
| hipSYCL    | [`HEAD`](https://github.com/illuhad/hipSYCL) (Clang 16, CUDA 12.1.0)\*                   | Ubuntu 23.04 | Debug, Release |

\* currently requires a patch for an illegal macro definition in CUDA:
  
```diff
--- a/include/crt/host_defines.h	2023-04-03 14:40:16.471254404 +0200
+++ b/include/crt/host_defines.h	2023-03-23 22:07:22.000000000 +0100
@@ -70,7 +70,7 @@
 #define __no_return__ \
         __attribute__((noreturn))
         
-#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)
+#if (defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)) && !defined(__clang__)
 /* gcc allows users to define attributes with underscores, 
    e.g., __attribute__((__noinline__)).
    Consider a non-CUDA source file (e.g. .cpp) that has the 

```