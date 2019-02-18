#!/bin/bash
set -e
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
pushd External/build/crossguid
cmake -DCMAKE_TOOLCHAIN_FILE=../../../cmake/Emscripten.cmake -DCMAKE_INSTALL_PREFIX=../../ ../../src/crossguid
cmake --build . --config debug --target install
popd

