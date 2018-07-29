@echo off
git submodule update --init External/src/zlib
mkdir External\build\zlib
pushd External\build\zlib
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=..\..\..\cmake\psp2snc.cmake -DCMAKE_INSTALL_PREFIX=../../ -G "NMake Makefiles" -DBUILD_SHARED_LIBS=off ../../src/zlib
cmake --build . --config release --target install
popd

