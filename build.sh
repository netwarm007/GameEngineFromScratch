#!/bin/bash
set -e
mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build . --config Debug
popd

