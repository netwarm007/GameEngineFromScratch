#!/bin/bash
set -e
git submodule update --init External/src/opengex
mkdir -p External/build/opengex
pushd External/build/opengex
CONFIG="${1:-Debug}"
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ "-DCMAKE_BUILD_TYPE=$CONFIG" ../../src/opengex
if [[ -z $1 ]];
then
    cmake --build . --config "$CONFIG" --target install
else
    cmake --build . --config "$CONFIG" --target install
fi
popd

