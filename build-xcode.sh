#!/bin/bash
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_BUILD_TYPE=Debug -G "Xcode" ..
cmake --build . --config debug

