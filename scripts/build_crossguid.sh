#!/bin/bash
set -e
git submodule update --init External/src/crossguid
mkdir -p External/build/crossguid
pushd External/build/crossguid
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ ../../src/crossguid
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

