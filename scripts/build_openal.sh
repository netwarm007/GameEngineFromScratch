#!/bin/bash
set -e
git submodule update --init External/src/openal
mkdir -p External/build/openal
pushd External/build/openal
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ ../../src/openal
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi
popd
