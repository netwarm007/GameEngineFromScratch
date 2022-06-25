#!/bin/bash
set -e
git submodule update --init External/src/ISPCTextureCompressor
mkdir -p External/build/ISPCTextureCompressor
cd External/build/ISPCTextureCompressor
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ ../../src/ISPCTextureCompressor
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi

