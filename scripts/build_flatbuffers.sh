#!/bin/bash
set -e
git submodule update --init External/src/flatbuffers
mkdir -p External/build/flatbuffers
pushd External/build/flatbuffers
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ ../../src/flatbuffers
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

