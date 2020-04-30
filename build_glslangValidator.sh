#!/bin/bash
set -e
git submodule update --init External/src/glslang
mkdir -p External/build/glslang
pushd External/build/glslang
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DBUILD_EXTERNAL=NO ../../src/glslang
cmake --build . --config Release --target install
popd
