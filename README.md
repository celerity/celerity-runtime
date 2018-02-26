# CELERITY Full Stack Example

This is a "full stack example" that both acts as a showcase and proof-of-concept
prototype of the most important aspects of the future CELERITY distributed
runtime implementation.

## Dependencies

* [Boost](http://www.boost.org) (tested with version 1.66.0)
* The [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp) runtime (tested with version 0.5.1)
* [CMake](https://www.cmake.org)
* A C++14 compiler (tested only with MSVC 14 so far)

### Optional
These dependencies are only required for live plotting of graphs.

* [NodeJS](https://nodejs.org/en)
* [GraphViz](http://graphviz.org)

## Building (on Windows)

	mkdir build
	cd build
	cmake -G "Visual Studio 14 2015 Win64" \
		-DCOMPUTECPP_PACKAGE_ROOT_DIR="<path to ComputeCpp installation>" \
		-DBOOST_ROOT="<path to boost library>" \
		..

## Running with Live Graph plotting (on Windows)
Assuming a debug binary is located in the `build` folder, simply run

	node liveplot.js

To continuously re-run the executable whenever it is updated, run

	node liveplot.js --watch

The `view_graphs.html` file can be used to display the newest graphs generated
by `liveplot.js` automatically.
