![Celerity Logo](docs/celerity_logo.png)

# Celerity Runtime - [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/celerity/celerity-runtime/blob/master/LICENSE) [![Semver 2.0](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/spec/v2.0.0.html) [![PRs # Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/celerity/celerity-runtime/blob/master/CONTRIBUTING.md)

The Celerity distributed runtime and API aims to bring the power and ease of
use of SYCL to distributed memory clusters.

**NOTE**: Celerity is a research project first and foremost, and is still in
early development. While it does work for certain applications, it probably
does not fully support your use case just yet. We'd however love for you to
give it a try and tell us about how you could imagine using Celerity for your
projects in the future.

## Dependencies

* A supported SYCL implementation, either
	- [hipSYCL](https://github.com/illuhad/hipsycl), or
	- [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp)
* [Boost](http://www.boost.org) (tested with version 1.65 - 1.68)
* A MPI 2 implementation (tested with OpenMPI 4.0 and MSMPI 10.0)
* [CMake](https://www.cmake.org)
* A C++14 compiler

## Building

Building can be as simple as calling `cmake && make`, depending on your setup
you might however also have to provide some library paths etc.

The runtime comes with several examples that are built automatically when
the `CELERITY_BUILD_EXAMPLES` CMake option is set (true by default).

## Using Celerity as a Library

Simply run `make install` (or equivalent, depending on build system) to copy all
relevant header files and libraries to the `CMAKE_INSTALL_PREFIX`. This includes
a CMake [package configuration
file](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#package-configuration-file)
which is placed inside the `lib/cmake` directory. Once included in a CMake
project, you can use the `add_celerity_to_target(TARGET target SOURCES source1
source2...)` function to set everything up.

## Running a Celerity Application

Celerity is built on top of MPI, which means a Celerity application can be
executed like any other MPI application (i.e., using `mpirun` or equivalent).

## Environment Variables

* `CELERITY_DEVICES` can be used to assign different compute devices to CELERITY
  nodes on a single host. The syntax is as follows:
  `CELERITY_DEVICES="<platform_id> <first device_id> <second device_id> ... <nth device_id>"`.
* `CELERITY_FORCE_WG=<work_group_size>` can be used to force a particular work
   group size for *every kernel* and *every dimension*.
* `CELERITY_PROFILE_OCL` controls whether OpenCL-level profiling information
  should be used or not (currently not supported when using hipSYCL).
* `CELERITY_LOG_LEVEL` controls the logging output level. One of `trace`, `debug`,
  `info`, `warn`, `err`, `critical`, or `off`.

