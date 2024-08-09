---
id: installation
title: Installation
sidebar_label: Installation
---

Celerity can be built and installed from
[source](https://github.com/celerity/celerity-runtime) using
[CMake](https://cmake.org). It requires the following dependencies:

- A MPI 2 implementation (for example [OpenMPI 4](https://www.open-mpi.org))
- A C++17 compiler
- A supported SYCL implementation (see below)

Note that while Celerity does support compilation and execution on Windows in
principle, in this documentation we focus exclusively on Linux, as it
represents the de-facto standard in HPC nowadays.

## Picking a SYCL Implementation

Celerity currently supports two different SYCL implementations. If you're
simply giving Celerity a try, the choice does not matter all that much. For
more advanced use cases or specific hardware setups it might however make
sense to prefer one over the other.

### AdaptiveCpp

[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) is an open source SYCL
and C++ standard parallelism implementation focused on leveraging existing toolchains
such as CUDA or HIP, making it a great choice when directly targeting Nvidia CUDA
and AMD ROCm platforms.

> AdaptiveCpp is currently available on Linux and has experimental/partial support
> for OSX and Windows.

### DPC++

Intel's LLVM fork [DPC++](https://github.com/intel/llvm) brings SYCL to the
latest Intel CPU and GPU hardware and also, experimentally, to CUDA and HIP
devices. Celerity will automatically detect when `CMAKE_CXX_COMPILER` points to
a DPC++ Clang.

To launch kernels on Intel GPUs, you will also need to install a recent version of the
[Intel Compute Runtime](https://github.com/intel/compute-runtime/releases) (failing to do so will
result in mysterious segfaults in the DPC++ SYCL library!)

> Celerity works with DPC++ on Linux.

Until its discontinuation in July 2023, Celerity also supported ComputeCpp as a SYCL implementation.

## Configuring CMake

After installing all of the aforementioned dependencies, clone (we recommend
using `git clone --recurse-submodules`) or download
the [Celerity source files](https://github.com/celerity/celerity-runtime) from GitHub. Next, create
a `build` folder inside the Celerity root folder and navigate into it.

The exact CMake configuration call you need depends on a few factors, for example the SYCL
implementation you chose, as well as your target hardware
platform. Here are a couple of examples:

<!--DOCUSAURUS_CODE_TABS-->

<!-- AdaptiveCpp + Ninja -->

```
cmake -G Ninja .. -DCMAKE_PREFIX_PATH="<path-to-acpp-install>" -DACPP_TARGETS="cuda:sm_52" -DCMAKE_INSTALL_PREFIX="<install-path>" -DCMAKE_BUILD_TYPE=Release
```

<!-- DPC++ + Unix Makefiles-->

```
cmake -G "Unix Makefiles" .. -DCMAKE_CXX_COMPILER="/path/to/dpc++/bin/clang++" -DCMAKE_INSTALL_PREFIX="<install-path>" -DCMAKE_BUILD_TYPE=Release
```

<!--END_DOCUSAURUS_CODE_TABS-->

In case multiple SYCL implementations are in CMake's search path, you can disambiguate them
using `-DCELERITY_SYCL_IMPL=AdaptiveCpp|DPC++`.

Note that the `CMAKE_PREFIX_PATH` parameter should only be required if you
installed SYCL in a non-standard location. See the [CMake
documentation](https://cmake.org/documentation/) as well as the documentation
for your SYCL implementation for more information on the other parameters.

Celerity comes with several example applications that are built by default.
If you don't want to build examples, provide `-DCELERITY_BUILD_EXAMPLES=0` as
an additional parameter to your CMake configuration call.

Celerity supports runtime and application profiling with [Tracy](https://github.com/wolfpld/tracy).
The integration disabled by default, build it with `-DCELERITY_TRACY_SUPPORT=1`. At runtime,
it must then be enabled with the `CELERITY_TRACY` environment variable (see [README](../README.md)).

## Building and Installing

After you have successfully configured CMake, building and installing
Celerity should be as simple as calling `ninja install` or `make install`.

If you just want to run the examples, you can skip the installation and
simply call `ninja` or `make`.

## Running Examples

If you have configured CMake to build the Celerity example applications, you
can now run them from within the build directory. For example, try running:

```
mpirun -n 2 ./examples/matmul/matmul
```

> **Tip:** You might also want to try and run the unit tests that come with Celerity.
> To do so, simply run `ninja test` or `make test`.

## Bootstrap your own Application

All projects in the `examples/` directory are stand-alone Celerity programs
– if you like a template for getting started, just copy one of them to
bootstrap on your own Celerity application. You can find out more about that
[here](https://github.com/celerity/celerity-runtime/blob/master/examples).
