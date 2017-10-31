#!/bin/bash
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config debug

