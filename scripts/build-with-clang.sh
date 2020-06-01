#!/bin/bash
set -e
PATH=$(pwd)/External/Linux/bin:$PATH
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/clang.cmake ..
cmake --build . --config debug

