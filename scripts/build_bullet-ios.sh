#!/bin/sh
set -e
git submodule update --init External/src/bullet
mkdir -p External/build/bullet
pushd External/build/bullet
rm -rf *
cmake -G Xcode -DPLATFORM=SIMULATOR64 -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios.toolchain.cmake -DCMAKE_INSTALL_PREFIX=../../iOS/ -DCMAKE_INSTALL_RPATH=../../iOS -DINSTALL_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF ../../src/bullet || exit 1
cmake --build . --config Release --target install
echo "Completed build of Bullet."
popd

