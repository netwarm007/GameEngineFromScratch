#!/bin/bash
set -e
git submodule update --init External/src/freetype
mkdir -p External/build/freetype
pushd External/build/freetype
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ ../../src/freetype
cmake --build . --config release --target install
popd

