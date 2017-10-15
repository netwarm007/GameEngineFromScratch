#!/bin/bash
git submodule update --init External/src/opengex
mkdir -p External/build/opengex
cd External/build/opengex
cmake -DCMAKE_INSTALL_PREFIX=../../Linux ../../src/opengex
cmake --build . --config release --target install

