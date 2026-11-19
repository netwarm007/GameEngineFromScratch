#!/bin/bash
set -e
git submodule update --init External/src/zlib
mkdir -p External/build/zlib
pushd External/build/zlib
CONFIG="${1:-Debug}"
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ "-DCMAKE_BUILD_TYPE=$CONFIG" ../../src/zlib
if [[ -z $1 ]];
then
    cmake --build . --config "$CONFIG" --target install
else
    cmake --build . --config "$CONFIG" --target install
fi
popd

