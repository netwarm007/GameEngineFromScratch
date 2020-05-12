#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/External/$(uname -s)/lib
mkdir -p build
pushd build
cmake -G"Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug
popd

