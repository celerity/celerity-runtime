# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

We recommend using the following SYCL versions with this release:

- DPC++: ???
- hipSYCL++: ???
- SimSYCL++: ???

See our [platform support guide](docs/platform-support.md) for a complete list of all officially supported configurations.

### Added

- Add support for SimSYCL as a SYCL implementation (#238)
- Extend compiler support to GCC (optionally with sanitizers) and C++20 code bases (#238)

## [0.5.0] - 2023-12-21

We recommend using the following SYCL versions with this release:

- DPC++: 61e51015 or newer
- hipSYCL: d2bd9fc7 or newer

### Added

- Add new environment variable `CELERITY_PRINT_GRAPHS` to control whether task and command graphs are logged (#197, #236)
- Introduce new experimental `for_each_item` utility to iterate over a celerity range (#199)
- Add new environment variables `CELERITY_HORIZON_STEP` and `CELERITY_HORIZON_MAX_PARALLELISM` to control Horizon generation (#199)
- Add support for out-of-bounds checking for host accessors (also enabled via `CELERITY_ACCESSOR_BOUNDARY_CHECK`) (#211)
- Add new `debug::set_task_name` utility for naming tasks to aid debugging (#213)
- Add new `experimental::constrain_split` API to limit how a kernel can be split (#212)
- Add GDB pretty-printers for common Celerity types (#207)
- `distr_queue::fence` and `buffer_snapshot` are now stable, subsuming the `experimental::` APIs of the same name (#225)
- Celerity now warns at runtime when a task declares reads from uninitialized buffers or writes with overlapping ranges between nodes (#224)
- Introduce new `experimental::hint` API for providing the runtime with additional information on how to execute a task (#227)
- Introduce new `experimental::hints::split_1d` and `experimental::hints::split_2d` task hints for controlling how a task is split into chunks (#227)

### Changed

- Horizons can now also be triggered by graph breadth. This improves performance in some scenarios, and prevents programs with many independent tasks from running out of task queue space (#199)

### Fixed

- In edge cases, command graph generation would fail to generate await-push commands when re-distributing reduction results (#223)
- Command graph generation was missing an anti-dependency between push-commands of partial reduction results and the final reduction command (#223)
- Don't create multiple smaller push-commands instead of a single large one in some rare situations (#229)
- Unit tests that inspect logs contained a race that would cause spurious failures (#234)

### Internal

- Improve command graph testing infrastructure (#198)
- Overhaul internal grid region and box representation, remove AllScale dependency (#204)

## [0.4.1] - 2023-09-08

We recommend using the following SYCL versions with this release:

- DPC++: 61e51015 or newer
- hipSYCL: d2bd9fc7 or newer

See our [platform support guide](docs/platform-support.md) for a complete list of all officially supported configurations.

### Removed

- Remove outdated workarounds for unsupported SYCL versions (#200, 85b7479c)

### Fixed

- Fix the behavior of dry runs (`CELERITY_DRY_RUN_NODES`) in the presence of fences or graph horizons (#196, 069f5029)
- Compatibility with recent hipSYCL >= d2bd9fc7 (#200, b174df7d)
- Compatibility with recent versions of Intel oneAPI and Arc-series dedicated GPUs (requires deactivating mimalloc, #203, c1519624)
- Work around a [bug in DPC++](https://github.com/intel/llvm/issues/10982) that breaks selection of the non-default device (#210, 2b652f8)

## [0.4.0] - 2023-07-13

We recommend using the following SYCL versions with this release:

- DPC++: 61e51015 or newer
- hipSYCL: 24980221 or newer

See our [platform support guide](docs/platform-support.md) for a complete list of all officially supported configurations.

### Added

- Introduce new experimental `host_object` and `side_effect` APIs to express non-buffer dependencies between host tasks (#68, 7a5326a)
- Add new `CELERITY_GRAPH_PRINT_MAX_VERTS` config options (#80, d3dd722)
- Named threads for better debugging (#98, 25d769d, #131, ff5fbec)
- Add support for passing device selectors to distr_queue constructor (#113, 556b6f2)
- Add new `CELERITY_DRY_RUN_NODES` environment variable to simulate the scheduling of an application on a large number of nodes (without execution or data transfers) (#125, 299ebbf)
- Add ability to name buffers for debugging (#132, 1076522)
- Introduce experimental `fence` API for accessing buffer and host-object data from the main thread (#151, 6b803f8)
- Introduce backend system for vendor-specific code paths (#162, 750f32a)
- Add `CELERITY_USE_MIMALLOC` CMake configuration option to use the mimalloc allocator (enabled by default) (#170, 234e3d2)
- Support 0-dimensional buffers, accessors and kernels (#163, 0685d94)
- Introduce new diagnostics utility for detecting erroneous reference captures into kernel functions, as well as unused accessors (#173, ff7ed02)
- Introduce `CELERITY_ACCESSOR_BOUNDARY_CHECK` CMake option to detect out-of-bounds buffer accesses inside device kernels (enabled by default for debug builds) (#178, 2c738c8)
- Print more helpful error message when buffer allocations exceed available device memory (#179, 79f97c2)

### Changed

- Update spdlog to 1.9.2 (#80, a178828)
- Overhaul logging mechanism (#80, 1b19bfc)
- Improve graph dependency tracking performance (#100, c9dab18)
- Improve task lookup performance (#112, 5139256)
- Introduce epochs as a mechanism for in-graph synchronization (#86, 61dd07e)
- Miscellaneous performance improvements (#115, 9a099d2, #137, b0254fd, #138, 02258c0, #145, f0b53ce)
- Improve scheduler performance by reducing lock contention (#111, 4547b5f)
- Improve graph generation and printing performance (#133, 8122798)
- Use [libenvpp](https://github.com/ph3at/libenvpp) to validate all `CELERITY_*` environment variables (#158, b2ced9b)
- Use native ("USM") pointers instead of SYCL buffers for backing buffer allocations (#162, 44497b3)
- Implement `range` and `id` types instead of aliasing SYCL types (#163, 0685d94)
- Disallow in-source builds (#176, 0a96d15)
- Lift restrictions on reductions for DPC++ (#175, efff21b)
- Remove multi-pass mechanism to allow reference capture of buffers and host-objects into command group functions, in alignment with the SYCL 2020 API (#173, 0a743c7)
- Drastically improve performance of buffer data location tracking (#184, adff79e)
- Switch to distributed scheduling model (#186, 0970bff)

### Deprecated

- Passing `sycl::device` to `distr_queue` constructor (use a device selector instead) (#113, 556b6f2)
- Capturing buffers and host objects by value into command group functions (capture by reference instead) (#173, 0a743c7)
- `allow_by_ref` is no longer required to capture references into command group functions (#173, 0a743c7)

### Removed

- Removed support for ComputeCpp (discontinued) (#167, 68367dd)
- Removed deprecated `host_memory_layout` (use `buffer_allocation_window` instead) (#187, f5e6510)
- Removed deprecated kernel dimension template parameter on `one_to_one`, `fixed` and `all` range mappers (#187, 40a12a4)
- Kernels can no longer receive `sycl::item` (use `celerity::item` instead), this was already broken in 0.3.2 (#163, 67ccacc)

### Fixed

- Improve performance for buffer transfers on IBM Spectrum MPI (#114, c60527f)
- Increase size limit on individual buffer transfer operations from 2 GiB to 128 GiB (#153, 972682f)
- Fix race between creating collective groups and submitting host tasks (#152, 0a4fca5)
- Align read-accessor `operator[]` with SYCL 2020 spec by returning const-reference instead of value (#156, 5011ded)

### Internal

- Add microbenchmark suite (#100, c2853ca, #107, 51f5bc5)
- Update Catch2 to v3.3 (#102, 9a6f19d, #129, 0d1e36a, #162, 5aa33d6)
- Add all_tests unit test executable (#104, c12b052)
- Add custom CSV and Markdown reporters (#109, ba3af8b)
- Introduce automatic clang-tidy checks for CI (#128, ca94bee)

## [0.3.2] - 2022-02-17

### Added

- Add support for ComputeCpp 2.7.0 and 2.8.0 with stable and experimental compilers. (2831b2a)
- Add support for using local memory with ComputeCpp. (8e2fce4)
- Print Celerity version upon runtime startup. (0681c16)
- Print warning when too few logical cores are available. (113e688)

### Fixed

- Fix race condition around reference-capture in matmul example. (76f49c9)
- Reduce hardware requirements for maximum work-group size in tests. (008a868, f0cf3f42)
- Update Catch2 submodule to v2.13.8 as a [bugfix](https://github.com/catchorg/Catch2/issues/2178). (26ca0895)
- Do not create empty chunks when splitting tasks with a small execution range in dimension 0. (15fa9293)
- Correctly handle empty buffers and buffer requirements with empty ranges. (ad99522b)
- Suppress unhelpful deprecation warnings around `sycl::atomic` from DPC++. (39dacdf5)
- Throw when submitting compute tasks with an empty execution range instead of accepting SYCL misbehavior. (baa242ad)

## [0.3.1] - 2022-01-04

We recommend Clang >= 10.0 as the host compiler to avoid false-positive
deprecation warnings.

### Fixed

- Remove blanket-statement error message upon early buffer deallocation, which
  in many cases is now legal. (6851145)
- Properly apply horizons to collective host task order-dependencies. (4488724)
- Avoid race between horizon task generation and horizon command execution.
  (f670868)
- Fix data race in `task_manager::notify_horizon_executed` (only in debug
  builds). (f641bcb)
- Don't rely on static destruction order for `user_benchmarker`. (d1c9e51)
- Restructure `wave_sim` example to avoid host side race condition for certain
  `--sample-rate` configurations. (d226b95)
- Hard-code paths for CMake dependencies in installed Celerity config to avoid
  mismatches. (4e88657)

## [0.3.0] - 2021-11-16

We recommend using the following SYCL versions with this release:

- ComputeCpp: 2.6.0 (an earlier version of this document used to recommend
  "2.6.0 or newer")
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
