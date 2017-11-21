#!/bin/bash
mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug
popd

