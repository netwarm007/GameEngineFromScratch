#!/bin/bash
set -e
git submodule update --init External/src/zlib
mkdir -p External/build/zlib
pushd External/build/zlib
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ ../../src/zlib
cmake --build . --config Release --target install
popd

