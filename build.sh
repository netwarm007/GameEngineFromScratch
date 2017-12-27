#!/bin/bash
mkdir -p build
pushd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/android.cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug
popd

