#!/bin/bash
set -e
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
pushd External/build/crossguid
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ ../../src/crossguid
cmake --build . --config Release --target install
popd

