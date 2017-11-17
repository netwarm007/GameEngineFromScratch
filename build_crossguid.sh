#!/bin/bash
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
cd External/build/crossguid
cmake -DCMAKE_INSTALL_PREFIX=../../ ../../src/crossguid
cmake --build . --config debug --target install

