#!/bin/bash
set -e
git submodule update --init External/src/ispc
mkdir -p External/build/ispc
cd External/build/ispc
export INSTALL_ROOT=../../`uname -s`
export LLVM_HOME=../../src/llvm
export LLVM_VERSION=LLVM_8_0
export ISPC_HOME=../../src/ispc
export PATH=$INSTALL_ROOT/bin:$PATH
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_ROOT -DARM_ENABLED=ON -DNVPTX_ENABLED=OFF -DISPC_INCLUDE_EXAMPLES=OFF -DBISON_EXECUTABLE=$INSTALL_ROOT/bin/bison $ISPC_HOME
make -j4
make install

