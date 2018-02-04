#!/bin/bash
set -e
git submodule update --init External/src/opengex
mkdir -p External/build/opengex
pushd External/build/opengex
cmake -DCMAKE_TOOLCHAIN_FILE=../../../cmake/android.cmake -DCMAKE_INSTALL_PREFIX=../../ ../../src/opengex
cmake --build . --config release --target install
popd

