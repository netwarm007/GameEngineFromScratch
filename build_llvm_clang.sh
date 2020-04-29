#!/bin/bash
set -e
git submodule update --init External/src/llvm
mkdir -p External/build/llvm
pushd External/build/llvm
cmake -G "Ninja" -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON -DCMAKE_INSTALL_PREFIX=../../$(uname -s) -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;ARM;X86;AArch64;NVPTX" -DLLVM_ENABLE_PROJECTS="clang" ../../src/llvm/llvm
cmake --build . --config Release --target install
popd

