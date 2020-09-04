---
id: installation
title: Installation
sidebar_label: Installation
---

Celerity can be built and installed from
[source](https://github.com/celerity/celerity-runtime) using
[CMake](https://cmake.org). It requires the following dependencies:

- [Boost](https://boost.org) (**Note**: There appear to be some issues with
  newer versions of Boost - pending investigation. In the meantime we
  recommend using a version ranging from 1.65 to 1.68)
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

### hipSYCL

[hipSYCL](https://github.com/illuhad/hipsycl) is an open source SYCL
implementation based on AMD HIP. While not fully spec-conformant (especially
regarding its OpenCL interoperability, which is fundamentally incompatible
with its design), hipSYCL is a great choice when directly targeting Nvidia
CUDA and AMD ROCm platforms.

> hipSYCL is currently only available on Linux.

### ComputeCpp

Codeplay's ComputeCpp is a fully SYCL 1.2.1 spec-conformant proprietary
implementation. Binary distributions can be downloaded from [Codeplay's
website](https://www.codeplay.com/products/computesuite/computecpp).
You will also need the CMake module from the [ComputeCpp
SDK](https://github.com/codeplaysoftware/computecpp-sdk/).

> ComputeCpp is available for both Linux and Windows.

## Configuring CMake

After installing all of the aforementioned dependencies, clone (we recommend
using `git clone --recurse-submodules`) or download the [Celerity source
files](https://github.com/celerity/celerity-runtime) from GitHub. Next,
create a `build` folder inside the Celerity root folder and navigate into it.

The exact CMake configuration call you need depends on a few factors, for
example the SYCL implementation you chose, as well as your target hardware
platform. Here are a couple of examples:

<!--DOCUSAURUS_CODE_TABS-->

<!--hipSYCL + Ninja -->

```
cmake -G Ninja .. -DCMAKE_PREFIX_PATH="<path-to-hipsycl-install>/lib" -DHIPSYCL_PLATFORM=cuda -DHIPSYCL_GPU_ARCH=sm_52 -DCMAKE_INSTALL_PREFIX="<install-path>" -DCMAKE_BUILD_TYPE=Release
```

<!--ComputeCpp + Unix Makefiles-->

```
cmake -G "Unix Makefiles" .. -DComputeCpp_DIR="<path-to-computecpp-install>" -DCMAKE_INSTALL_PREFIX="<install-path>" -DCMAKE_BUILD_TYPE=Release
```

<!--END_DOCUSAURUS_CODE_TABS-->

Note that the `CMAKE_PREFIX_PATH` and `ComputeCpp_DIR` parameters should only
be required if you installed SYCL in a non-standard location. See the [CMake
documentation](https://cmake.org/documentation/) as well as the documentation
for your SYCL implementation for more information on the other parameters.

> We currently recommend using the [Ninja build
> system](https://ninja-build.org/) for building hipSYCL-based projects due
> to some issues with dependency tracking that CMake has with Unix Makefiles.

Celerity comes with several example applications that are built by default.
If you don't want to build examples, provide `-DCELERITY_BUILD_EXAMPLES=0` as
an additional parameter to your CMake configuration call.

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
