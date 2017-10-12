#!/bin/bash
git submodule update --init
mkdir -p build/llvm
cd build/llvm
cmake -DCMAKE_INSTALL_PREFIX=../../Linux -DLLVM_LIBDIR_SUFFIX=64 -DLLVM_ENABLE_PROJECTS=clang ../../src/llvm
cmake --build . --config release --target install

