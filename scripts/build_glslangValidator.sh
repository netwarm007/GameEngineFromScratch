#!/bin/bash
set -e
git submodule update --init External/src/glslang
mkdir -p External/build/glslang
pushd External/build/glslang
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DBUILD_EXTERNAL=NO ../../src/glslang
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd
