#!/bin/bash
mkdir -p build
cd build
rm -rf *
cmake ..
cmake --build . --config debug

