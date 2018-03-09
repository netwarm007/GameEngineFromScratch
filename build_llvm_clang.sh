#!/bin/bash
set -e
git submodule update --init External/src/llvm External/src/clang
mkdir -p External/build/llvm
pushd External/build/llvm
cmake -DLLVM_ENABLE_DUMP=ON -DCMAKE_INSTALL_PREFIX=../../Linux -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang -DLLVM_PARALLEL_COMPILE_JOBS=8 ../../src/llvm
make -j8
make install
popd

