#!/bin/sh
set -e
git submodule update --init External/src/bullet
mkdir -p External/build/bullet
pushd External/build/bullet
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DBUILD_SHARED_LIBS=ON -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE ../../src/bullet
cmake --build . --config release --target install
popd

