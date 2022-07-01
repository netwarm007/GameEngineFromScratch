#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/External/$(uname -s)/lib
mkdir -p build
pushd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_OSX_ARCHITECTURES=x86_64 -G "Xcode" ..
if [[ -z $1 ]];
then
    cmake --build . --config Debug
else
    cmake --build . --config $1
fi
popd


