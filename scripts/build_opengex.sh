#!/bin/bash
set -e
git submodule update --init External/src/opengex
mkdir -p External/build/opengex
pushd External/build/opengex
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ ../../src/opengex
cmake --build . --config Release --target install
popd

