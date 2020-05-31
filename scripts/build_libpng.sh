#!/bin/bash
set -e
git submodule update --init External/src/libpng
mkdir -p External/build/libpng
cd External/build/libpng
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s` ../../src/libpng
cmake --build . --config Release --target install

