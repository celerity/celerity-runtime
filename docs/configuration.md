---
id: configuration
title: Configuration
sidebar_label: Configuration
---

After successfully [installing](installation.md) Celerity, you can tune its runtime behaviour via a number of environment variables. This page lists all available options.

Note that same of these runtime options require a [corresponding CMake option](installation.md#additional-configuration-options) to be enabled during the build process.
This is generally the case if the option has a negative impact on performance, or if it is not required in most use cases.

Celerity uses [libenvpp](https://github.com/ph3at/libenvpp) for environment variable handling,
which means that typos in environment variable names or invalid values will be detected and reported.

## Environment Variables for Debugging and Profiling

The following environment variables can be used to control Celerity's runtime behaviour,
specifically in development, debugging, and profiling scenarios:
| Option | Values | Description |
| --- | --- | --- |
| `CELERITY_LOG_LEVEL` | `trace`, `debug`, `info`, `warn`, `err`, `critical`, `off` | Controls the logging output level. |
| `CELERITY_PROFILE_KERNEL` | `on`, `off` | Controls whether SYCL queue profiling information should be queried. This typically incurs additional overhead for each kernel launch. |
| `CELERITY_PRINT_GRAPHS` | `on`, `off` | Controls whether task, command and instruction graphs are logged in Graphviz format at the end of execution (requires log level `info` or higher). Note that these can quickly become quite large, even for small applications. |
| `CELERITY_DRY_RUN_NODES` | *number* | Simulates a run with the given number of nodes without actually executing any instructions (allocations, kernels, host tasks, etc). Useful for investigating performance characteristics of the runtime itself. |
| `CELERITY_TRACY` | `off`, `fast`, `full` | Controls the Tracy profiler integration. Set to `off` to disable, `fast` for light integration with little runtime overhead, and `full` for integration with extensive performance debug information included in the trace. Only available if integration was enabled enabled at build time through the CMake option `-DCELERITY_TRACY_SUPPORT=ON`.

## Environment Variables for Performance Tuning

The following environment variables can be used to tune Celerity's performance. 
Generally, these might need to be changed depending on the specific application and hardware setup to achieve the best possible performance, but the default values should work reasonably well in all cases:
| Option | Values | Description |
| --- | --- | --- |
| `CELERITY_HORIZON_STEP` | *number* | Determines the maximum number of sequential tasks before a new [horizon task](https://doi.org/10.1007/s42979-024-02749-w) is introduced. |
| `CELERITY_HORIZON_MAX_PARALLELISM` | *number* | Determines the maximum number of parallel tasks before a new horizon task is introduced. |
| `CELERITY_BACKEND_DEVICE_SUBMISSION_THREADS` | `on`, `off` | Controls whether device commands are submitted in a separate backend thread for each local device. This improves performance particularly in cases where kernel runtimes are very short. (default: `on`) |
| `CELERITY_THREAD_PINNING` | `off`, `auto`, `from:#`, *core list* | Controls if and how threads are pinned to CPU cores. `off` disables pinning, `auto` lets Celerity decide, `from:#` starts pinning sequentially from the given core, and a core list specifies the exact pinning (see below). (default: `auto`) |

### Notes on Core Pinning

Some Celerity threads benefit greatly from rapid communication, and we have observed performance differences of up to 50% for very fine-grained applications when pinning threads to specific CPU cores. The `CELERITY_THREAD_PINNING` environment variable can be set to a list of CPU cores to which Celerity should pin its threads.

In most cases, `auto` should provide very good results. However, if you want to manually specify the core list, you can do so by providing a comma-separated list of core numbers. The length of this list needs to precisely match the number of threads and order which Celerity is using -- for detailed information consult the definition of `celerity::detail::thread_pinning::thread_type`.
