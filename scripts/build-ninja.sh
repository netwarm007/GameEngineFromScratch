#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/External/$(uname -s)/lib
mkdir -p build
pushd build
# CMAKE_ARGS is intentionally word-split into extra cmake options.
cmake -G"Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $CMAKE_ARGS ..
if [[ -z $1 ]];
then
    cmake --build . --config Debug
else
    cmake --build . --config $1
fi
popd

