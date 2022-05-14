#!/bin/bash
set -e
git submodule update --init External/src/opengex
mkdir -p External/build/opengex
pushd External/build/opengex
cmake -G Xcode -DPLATFORM=SIMULATOR64 -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DCMAKE_INSTALL_PREFIX=../../ ../../src/opengex
cmake --build . --config release --target install
popd

