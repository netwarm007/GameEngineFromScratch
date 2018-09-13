#!/bin/bash
set -e
git submodule update --init External/src/glslang
mkdir -p External/build/glslang
cd External/build/glslang
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/glslang || exit 1
cmake --build . --config release --target install
echo "Completed build of glslangValidator"
