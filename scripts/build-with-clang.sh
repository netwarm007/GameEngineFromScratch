#!/bin/bash
set -e
PATH=$(pwd)/External/Linux/bin:$PATH
mkdir -p build
pushd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/clang.cmake ..
if [[ -z $1 ]];
then
    cmake --build . --config Debug
else
    cmake --build . --config $1
fi
popd

