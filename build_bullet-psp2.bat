@echo off
git submodule update --init External\src\bullet
mkdir External\build\bullet
pushd External\build\bullet
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=../../../cmake/psp2snc.cmake -DCMAKE_INSTALL_PREFIX=..\..\PSP2\ -DCMAKE_INSTALL_RPATH=..\..\PSP2\ -DINSTALL_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_BULLET2_DEMOS=OFF -DBUILD_BULLET3=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -G "NMake Makefiles" ..\..\src\bullet
cmake --build . --config release --target install
popd
