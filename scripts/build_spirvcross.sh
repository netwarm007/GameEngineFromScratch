#!/bin/bash
set -e
git submodule update --init External/src/spirv-cross
mkdir -p External/build/spirv-cross
pushd External/build/spirv-cross
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ ../../src/spirv-cross
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi
popd
