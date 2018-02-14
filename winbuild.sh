#!/bin/sh
mkdir -p build
cd build
/mnt/c/Users/Philip\ Salzmann/Desktop/TOOLS/cmake-3.10.2-win64-x64/bin/cmake.exe \
	-G "Visual Studio 14 2015 Win64" \
	-DCOMPUTECPP_PACKAGE_ROOT_DIR="C:/Program Files/Codeplay/ComputeCpp" \
	-DBOOST_ROOT="C:/psalz_cxx_libs/boost_1_66_0" \
	.. \
	|| echo "ATTENTION: It may be required to run cmake within the 'Developer Command Prompt for Visual Studio'"
