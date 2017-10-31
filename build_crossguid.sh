#!/bin/bash
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
cd External/build/crossguid
cmake -DCMAKE_INSTALL_PREFIX=../../Linux -DXG_TESTS=OFF ../../src/crossguid
cmake --build . --config release --target install

