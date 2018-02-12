#!/bin/sh
mkdir -p build
cd build
/mnt/c/Users/Philip\ Salzmann/Desktop/TOOLS/cmake-3.10.2-win64-x64/bin/cmake.exe \
	-DBOOST_ROOT="C:/psalz_cxx_libs/boost_1_66_0" \
	..
