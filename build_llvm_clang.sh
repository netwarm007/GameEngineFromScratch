#!/bin/bash
set -e
git submodule update --init External/src/llvm External/src/clang External/src/libcxx External/src/libcxxabi
mkdir -p External/build/llvm
pushd External/build/llvm
cmake -DLLVM_ENABLE_DUMP=ON -DCMAKE_INSTALL_PREFIX=../../$(uname -s) -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;ARM;X86;AArch64" -DLLVM_ENABLE_PROJECTS="clang" ../../src/llvm
make -j8
make install
popd

