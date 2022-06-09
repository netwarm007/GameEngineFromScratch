#!/bin/bash
set -e
git submodule update --init External/src/zlib
mkdir -p External/build/zlib
pushd External/build/zlib
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ ../../src/zlib
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

