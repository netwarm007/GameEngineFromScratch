#!/bin/bash
set -e
./scripts/build_ispc.sh
export PATH="$(pwd)/External/$(uname -s)/bin:$PATH"
git submodule update --init External/src/ISPCTextureCompressor
mkdir -p External/build/ISPCTextureCompressor
cd External/build/ISPCTextureCompressor
CONFIG="${1:-Debug}"
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../ "-DCMAKE_BUILD_TYPE=$CONFIG" ../../src/ISPCTextureCompressor
if [[ -z $1 ]];
then
    cmake --build . --config "$CONFIG" --target install
else
    cmake --build . --config "$CONFIG" --target install
fi

