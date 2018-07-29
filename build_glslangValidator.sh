#!/bin/bash
set -e
git submodule update External/src/glslangValidator
mkdir -p External/build/glslangValidator
cd External/build/glslangValidator
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/glslangValidator|| exit 1
cmake --build . --config release --target install
echo "Completed build of glslangValidator"
