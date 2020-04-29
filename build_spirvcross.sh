#!/bin/bash
set -e
git submodule update --init External/src/spirv-cross
mkdir -p External/build/spirv-cross
pushd External/build/spirv-cross
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ ../../src/spirv-cross
cmake --build . --config Release --target install
popd
