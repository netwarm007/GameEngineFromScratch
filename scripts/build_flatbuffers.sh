#!/bin/bash
set -e
git submodule update --init External/src/flatbuffers
mkdir -p External/build/flatbuffers
pushd External/build/flatbuffers
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ ../../src/flatbuffers
cmake --build . --config Release --target install
popd

