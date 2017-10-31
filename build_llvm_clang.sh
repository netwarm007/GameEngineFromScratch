#!/bin/bash
git submodule update --init External/src/llvm External/src/clang
mkdir -p External/build/llvm
cd External/build/llvm
cmake -DCMAKE_INSTALL_PREFIX=../../Linux -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang -DLLVM_PARALLEL_COMPILE_JOBS=8 ../../src/llvm
make -j8
make install

