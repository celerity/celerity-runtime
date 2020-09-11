# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

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

__Hello, World!__ This is our first release!
