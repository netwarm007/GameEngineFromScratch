#!/bin/bash
mkdir -p build
cd build
cmake -G "Visual Studio 15 2017 Win64" ..
cmake --build . --config debug

