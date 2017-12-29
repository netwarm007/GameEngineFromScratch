#!/bin/bash
git submodule update --init External/src/zlib
mkdir -p External/build/zlib
cd External/build/zlib
cmake -DCMAKE_INSTALL_PREFIX=../../ ../../src/zlib
cmake --build . --config release --target install

