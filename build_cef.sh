#!/bin/bash
set -e
git submodule update --init External/src/cef
mkdir -p External/build/cef
pushd External/build/cef
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ -DCMAKE_INSTALL_RPATH=../../ ../../src/cef
cmake --build . --config Release --target install
popd
