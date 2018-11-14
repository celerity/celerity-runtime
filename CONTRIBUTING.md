# Contributing

## Formatting Code Using Clang Format

All code should be formatted using `clang-format` before committing.
Unfortunately Clang Format frequently has breaking changes with major releases
(which are tied to the Clang release cycle), which means we have to settle on a
specific version.

We currently use clang-format 6.0. Please make sure to use the correct version
to avoid cluttering your diffs with unrelated format changes. [Prebuilt binaries
can be found here](http://releases.llvm.org/6.0.1/).

## Running Tests

Please verify that all tests are green (run with `make test` or `ctest`) and
consider adding your own tests if possible.

