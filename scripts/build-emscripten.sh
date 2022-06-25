#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/External/$(uname -s)/lib
mkdir -p build
pushd build
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=../cmake/Emscripten.cmake ..
if [[ -z $1 ]];
then
    cmake --build . --config Debug
else
    cmake --build . --config $1
fi
popd

