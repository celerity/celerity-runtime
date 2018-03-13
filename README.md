# CELERITY Full Stack Example

This is a "full stack example" that both acts as a showcase and proof-of-concept
prototype of the most important aspects of the future CELERITY distributed
runtime implementation.

## Dependencies

* [Boost](http://www.boost.org) (tested with version 1.66.0)
* The [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp)
  runtime (tested with version 0.6.1)
* [CMake](https://www.cmake.org)
* A C++14 compiler (tested only with MSVC 14 so far)

### Optional
These dependencies are only required for live plotting of graphs.

* [NodeJS](https://nodejs.org/en)
* [GraphViz](http://graphviz.org)

## Building (on Windows)

1. Prepare vcpkg

        git clone https://github.com/Microsoft/vcpkg.git
        cd .\vcpkg
        .\bootstrap-vcpkg.bat

2. Install dependencies using vcpkg (x64-windows triplet)

        .\vcpkg.exe install boost-graph:x64-windows
        .\vcpkg.exe install gtest:x64-windows

3. Integrate install

        .\vcpkg.exe integrate install

4. Create a copy of the `CMakeSettings.json.template` file and name it
   `CMakeSettings.json`

5. Use `File -> Open -> Folder` in Visual Studio 2017 to open the workspace

6. Right-click `CMakeLists.txt` in the Solution Explorer and choose "Change
   CMake Settings" to open the `CMakeSettings.json` file

7. Edit the `CMakeSettings.json` file by supplying the correct paths for finding
   the vcpkg CMake tool chain file (`VCPKG_ROOT`) and the install root of your
   ComputeCpp installation (`COMPUTECPP_PACKAGE_ROOT_DIR`) in the "environments"
   section:

```json
"environments": [
    {
        "COMPUTECPP_PACKAGE_ROOT_DIR": "<path-to-computecpp>",
        "VCPKG_ROOT": "<path-to-vcpkg>",
        ...
    }
]
```

8. Save file and use the top menu "CMake" to regenerate the cache.

9. Use the top menu "CMake" to build and/or run tests

10. Build artifacts will be placed in
    `<workspace_root>\\build\\\<configuration_name\>\\<build_type>` for example:
    + `<workspace_root>\\build\\\x64-Debug\\Debug`
    + `<workspace_root>\\build\\\x64-Release\\Release`

## Running with Live Graph plotting (on Windows)

Simply run

    node liveplot.js <path_to_exe>

To continuously re-run the executable whenever it is updated, run

    node liveplot.js <path_to_exe> --watch

The `view_graphs.html` file can be used to display the newest graphs generated
by `liveplot.js` automatically.
