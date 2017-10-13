#!/bin/bash
git submodule update --init
mkdir -p build/opengex
cd build/opengex
cmake -DCMAKE_INSTALL_PREFIX=../../Linux ../../src/opengex
cmake --build . --config release --target install

