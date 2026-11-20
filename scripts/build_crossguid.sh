#!/bin/bash
set -e
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
pushd External/build/crossguid
CONFIG="${1:-Debug}"
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ "-DCMAKE_BUILD_TYPE=$CONFIG" ../../src/crossguid
if [[ -z $1 ]];
then
    cmake --build . --config "$CONFIG" --target install
else
    cmake --build . --config "$CONFIG" --target install
fi
popd

