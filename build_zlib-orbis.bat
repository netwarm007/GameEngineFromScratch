@echo off
git submodule update --init External/src/zlib
mkdir External\build\zlib
pushd External\build\zlib
rm -rf *
cmake  -DCMAKE_TOOLCHAIN_FILE=..\..\..\cmake\orbis-clang.cmake -DCMAKE_INSTALL_PREFIX=../../ -G "NMake Makefiles" ../../src/zlib
cmake --build . --config release --target install
popd


