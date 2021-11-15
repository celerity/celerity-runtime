# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2021-11-16

We recommend using the following SYCL versions with this release:

- ComputeCpp: 2.6.0 or newer
- DPC++: 7735139 or newer
- hipSYCL: 7b00e2e or newer

### Added

- Introduce aliases or custom implementations in `::celerity` namespace for most
  supported SYCL features (e.g. `celerity::access_mode`, `celerity::item` and so
  on). (c36f55e, 44c181e, f7ca077, 2588d7e, fd58422, e706db3, 7552445)
- `CELERITY_PROFILE_KERNEL` is now also supported for hipSYCL. (0521e9e)
- Add support for unnamed kernels (hipSYCL and DPC++ only). (7a3431e)
- Add support for nd-range `parallel_for` alongside `local_accessor` (hipSYCL
  and DPC++ only) and `group` functions. (2588d7e)
- Add support for SYCL 2020-style reductions (hipSYCL with required patch, DPC++
  partially, ComputeCpp unsupported). (e79f765)
- Add support for DPC++ as a SYCL implementation. (44c181e)
- The API for accessing distributed data from within host tasks (previously
  known as `host_memory_layout`) is streamlined into a new
  `buffer_allocation_window` API. (ad66329)
- Add support for multi-dimensional subscript operators for host and device
  accessors. (c720ffb)
- Add support for SYCL 2020-style accessor constructors as well as new
  `access_mode` and `target` enums. (c36f55e)

### Changed

- Celerity will now fall back to using any available SYCL devices instead of
  aborting in case no GPUs can be found. (0d87461)
- Improve performance of some applications with large task graphs. (1417296)
- Rename `CELERITY_PROFILE_OCL` to `CELERITY_PROFILE_KERNEL`. (ae94ef4)
- Improve handling of configuration environment variables such as
  `CELERITY_DEVICES`. (fc595d2)
- Terminate execution in case an asynchronous SYCL error is reported. (10d3d75)
- Remove dependency on Boost. (ae66f22)
- Report an error when more than one build configuration (Debug, Release, ...)
  of Celerity is installed to the same location. (2c12944)

### Deprecated

- `host_memory_layout` and associated functions are being deprecated in favor of
  `buffer_allocation_window`. (ad66329)
- Range mappers no longer require kernel and buffer dimensions to be explicitly
  specified. Doing so will now trigger a deprecation warning. (50dd62d)

### Removed

- Remove support for `CELERITY_FORCE_WG` configuration environment variable.
  (afe81cc)

### Fixed

- Fix a crash occurring when creating new buffers after a number of tasks have
  already been processed. (d33c8d2)
- Forward correct offset / range to SYCL accessors internally. (349d58b)
- Fix a crash that could happen when building a Debug executable against a
  Release installation of Celerity or vice-versa. (2c12944)
- Fix a bug that could cause nested SYCL command groups to be submitted. (750b28a)

## [0.2.1] - 2020-09-09

### Fixed

- Re-enable ComputeCpp workaround for explicit copy operations. (f58b146b)
- Fix compilation on Windows by avoiding the `TRUE` literal as enum value.
  (8de922e9)
- Fix compilation with Boost < 1.67 by using backwards compatible header.
  (a51c98a7)

## [0.2.0] - 2020-09-04

### Added

- Following the release of the SYCL 2020 provisional spec, master access tasks
  have been retired in favor of host tasks. These are scheduled in command groups
  in the same fashion as compute tasks. In addition to master-only execution,
  they allow distributing host code amongst nodes. (bbf90637)
- Celerity buffers are now fully virtualized, meaning that your Celerity
  application only allocates as much memory as required on each node.
  (8a203872)
- _(Experimental)_ Collective host tasks allow integration of distributed I/O
  by providing an infrastructure to call MPI collective APIs like parallel
  HDF5 from within asynchronous Celerity tasks. (bbf90637)
- Properly support 3D kernels. (e5543bd0)

### Changed

- Celerity now uses (and requires) C++17. (5eec3e02)
- Celerity should now perform better with large command graphs. (5d876a5a)
- Celerity should now be able to automatically assign a unique compute device
  to each node on a host, given that a sufficient number of devices is
  available. (9d3da06e)

### Fixed

- Don't print an error message regarding buffer lifetime for trivial programs
  (i.e., programs containing no tasks). (ae133458)

### Removed

- Removed `celerity::queue::with_master_access`, which is replaced by the
  more powerful `celerity::handler::host_task`. (bbf90637)

## [0.1.0] - 2019-09-05

**Hello, World!** This is our first release!
