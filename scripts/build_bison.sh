#!/bin/bash
set -e
git submodule update --init External/src/bison
mkdir -p External/build/bison
cd External/build/bison
../../src/bison/configure --prefix=$(pwd)/../../`uname -s`/
make -j4
make install

