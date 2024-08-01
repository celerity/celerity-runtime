# Celerity Example Projects

These example projects are a useful starting point for developing your own Celerity
application. All examples will also be built automatically in-tree with the runtime
when the `CELERITY_BUILD_EXAMPLES` CMake option is set (true by default).

## Setup

Begin by copying an example project folder that best fits your use case:
```shell
cp -r celerity-runtime/examples/convolution my-project
```

## Build Configuration

Configure the project with CMake and make sure Celerity can be found by setting
[`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)
to include the Celerity installation directory. Depending on your SYCL implementation,
you may also have to specify additional target options.

### With hipSYCL (exemplary)

```shell
cd my-project
cmake -B build \
    -DCMAKE_PREFIX_PATH="../celerity-runtime-install" \
    -DHIPSYCL_TARGETS=cuda:sm_75
```

### With DPC++ (exemplary)

```shell
cd my-project
cmake -B build \
    -DCMAKE_PREFIX_PATH="../celerity-runtime-install" \
    -DCMAKE_CXX_COMPILER=/opt/dpcpp/bin/clang++
```
