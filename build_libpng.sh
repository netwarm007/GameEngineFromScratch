#!/bin/bash
git submodule update --init External/src/libpng
mkdir -p External/build/libpng
cd External/build/libpng
cmake -DCMAKE_INSTALL_PREFIX=../../Linux ../../src/libpng
cmake --build . --config release --target install

