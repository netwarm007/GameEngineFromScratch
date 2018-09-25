#!/bin/bash
set -e
git submodule update --init External/src/spirv-cross
mkdir -p External/build/spirv-cross
cd External/build/spirv-cross
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/spirv-cross || exit 1
cmake --build . --config release --target install
echo "Completed build of spirv-cross"
