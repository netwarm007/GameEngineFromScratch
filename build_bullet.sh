#!/bin/sh

git submodule update --init External/src/bullet
mkdir -p External/build/bullet
pushd External/build/bullet
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ../../src/bullet || exit 1
make -j $(command nproc 2>/dev/null || echo 12) || exit 1
make install || exit 1
echo "Completed build of Bullet."
popd

