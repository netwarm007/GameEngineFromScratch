#!/bin/sh
set -e
git submodule update --init External/src/bullet
mkdir -p External/build/bullet
cd External/build/bullet
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=../../../cmake/android.cmake -DCMAKE_INSTALL_PREFIX=../../Android/ -DCMAKE_INSTALL_RPATH=../../Android -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE ../../src/bullet || exit 1
cmake --build . --config release --target install
echo "Completed build of Bullet."

