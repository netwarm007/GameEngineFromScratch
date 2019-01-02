#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/External/$(uname -s)/lib
mkdir -p build
pushd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/Emscripten.cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build . --config Debug
popd

