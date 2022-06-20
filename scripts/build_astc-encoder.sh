#!/bin/bash
set -e
git submodule update --init External/src/astc-encoder
mkdir -p External/build/astc-encoder
pushd External/build/astc-encoder
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s` -DISA_AVX2=ON -DISA_SSE41=ON -DISA_SSE2=ON ../../src/astc-encoder
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

