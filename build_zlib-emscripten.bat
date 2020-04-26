@echo off
git submodule update --init External/src/zlib
mkdir External\build\zlib
pushd External\build\zlib
rm -rf *
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=../../../cmake/Emscripten.cmake -DCMAKE_INSTALL_PREFIX=../../ -G "NMake Makefiles" -DBUILD_SHARED_LIBS=off ../../src/zlib
cmake --build . --config Release --target install
popd

