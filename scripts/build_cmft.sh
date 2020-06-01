#!/bin/bash
set -e
git submodule update --init External/src/cmft
mkdir -p External/build/cmft
pushd External/build/cmft
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ ../../src/cmft
cmake --build . --config Release --target install
popd
