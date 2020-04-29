#!/bin/bash
set -e
git submodule update --init External/src/SDL
mkdir -p External/build/SDL
pushd External/build/SDL
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ ../../src/SDL
cmake --build . --config Release --target install
popd
