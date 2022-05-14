#!/bin/bash
set -e
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
pushd External/build/crossguid
cmake -G "Xcode" -DPLATFORM=SIMULATOR64 -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DCMAKE_INSTALL_PREFIX=../../ ../../src/crossguid
cmake --build . --config debug --target install
popd

