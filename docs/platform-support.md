---
id: platform-support
title: Officially Supported Platforms
sidebar_label: Platform Support
---

The most recent version of Celerity aims to support the following environments:

* hipSYCL ≥ revision [`d2bd9fc7`](https://github.com/illuhad/hipSYCL/commit/d2bd9fc7), with
  * CUDA ≥ 11.0
  * Clang ≥ 10.0 for CUDA &lt; 12.0, Clang ≥ 16.0 for CUDA ≥ 12.0
  * on NVIDIA hardware with compute capability ≥ 7.0
  * or on CPUs via OpenMP
* DPC++ ≥ revision [`89327e0a`](https://github.com/intel/llvm/commit/89327e0a)
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime) ≥ 23.22.26516.18
  * [oneAPI Level Zero](https://github.com/oneapi-src/level-zero) ≥ 1.9.9
  * on integrated and dedicated Intel GPUs
* SimSYCL [HEAD](https://github.com/celerity/SimSYCL)

ComputeCpp is no longer supported since its discontinuation.

## Continuously Tested Configurations

We automatically verify Celerity's build process and test suites against a select number of system configurations.

Those are (CRT = Intel Compute Runtime, L0 = oneAPI Level Zero):

| SYCL       | SYCL version                                                                                | OS           | GPU             | Build type     |
|------------|---------------------------------------------------------------------------------------------|--------------|-----------------|----------------|
| DPC++      | [`89327e0a`](https://github.com/intel/llvm/commit/89327e0a) (CRT 23.22.26516.18, L0 1.11.0) | Ubuntu 22.04 | Intel Arc 770   | Debug          |
| DPC++      | [`HEAD`](https://github.com/intel/llvm/) (CRT 23.22.26516.18, L0 1.11.0)                    | Ubuntu 22.04 | Intel Arc 770   | Debug, Release |
| hipSYCL    | [`d2bd9fc7`](https://github.com/illuhad/hipSYCL/commit/d2bd9fc7) (Clang 10, CUDA 11.0.3)    | Ubuntu 20.04 | NVIDIA RTX 2070 | Debug          |
| hipSYCL    | [`d2bd9fc7`](https://github.com/illuhad/hipSYCL/commit/d2bd9fc7) (Clang 14, CUDA 11.8.0)    | Ubuntu 22.04 | NVIDIA RTX 2070 | Debug, Release |
| hipSYCL    | [`HEAD`](https://github.com/illuhad/hipSYCL) (Clang 16, CUDA 12.2.0)\*                      | Ubuntu 23.04 | NVIDIA RTX 2070 | Debug, Release |
| SimSYCL    | [`HEAD`](https://github.com/celerity/SimSYCL) (GCC 11.4)                                    | Ubuntu 22.04 | (None)          | Debug, Release |

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